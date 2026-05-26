//! Custom Gemma model parser with optional Query/Key Normalization (Gemma 2 / Gemma 3 support).
//! Optimized and tracing-free for lightweight, dependency-lean CPU execution.

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::D;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::Module;
use candle_transformers::utils::repeat_kv;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn from_qtensor(qtensor: QTensor, eps: f64) -> Result<Self> {
        let weight = qtensor.dequantize(&Device::Cpu)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Calculate variance along the hidden dimension
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        // 2. Normalize x
        let x_norm = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        // 3. Apply the Gemma +1.0 weight offset
        let output = x_norm.broadcast_mul(&(self.weight.broadcast_add(&candle_core::Tensor::new(1.0f32, x.device())?)?))?;
        Ok(output)
    }
}

pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Clone)]
pub struct CustomEmbedding {
    raw_data: std::sync::Arc<[u8]>,
    ggml_dtype: candle_core::quantized::GgmlDType,
    vocab_size: usize,
    hidden_dim: usize,
    row_bytes: usize,
    layer_idx: Option<usize>,
    layer_bytes: usize,
}

impl std::fmt::Debug for CustomEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomEmbedding")
            .field("vocab_size", &self.vocab_size)
            .field("hidden_dim", &self.hidden_dim)
            .field("row_bytes", &self.row_bytes)
            .field("layer_idx", &self.layer_idx)
            .field("layer_bytes", &self.layer_bytes)
            .finish()
    }
}

impl CustomEmbedding {
    pub fn new(
        raw_data: Vec<u8>,
        ggml_dtype: candle_core::quantized::GgmlDType,
        vocab_size: usize,
        hidden_dim: usize,
        layer_idx: Option<usize>,
        total_cols: usize,
    ) -> Self {
        let block_size = ggml_dtype.block_size();
        let type_size = ggml_dtype.type_size();
        let row_bytes = (total_cols / block_size) * type_size;
        let layer_bytes = (hidden_dim / block_size) * type_size;
        Self {
            raw_data: std::sync::Arc::from(raw_data),
            ggml_dtype,
            vocab_size,
            hidden_dim,
            row_bytes,
            layer_idx,
            layer_bytes,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let (batch_size, seq_len) = xs.dims2()?;
        let flat_indices = xs.flatten_all()?.to_vec1::<u32>()?;
        
        let mut temp_raw_data = vec![0u8; flat_indices.len() * self.layer_bytes];
        for (i, &idx) in flat_indices.iter().enumerate() {
            let idx = idx as usize;
            if idx >= self.vocab_size {
                candle_core::bail!("index out of bounds in CustomEmbedding: {} >= {}", idx, self.vocab_size);
            }
            let src_start = match self.layer_idx {
                None => idx * self.row_bytes,
                Some(l) => idx * self.row_bytes + l * self.layer_bytes,
            };
            let dst_start = i * self.layer_bytes;
            temp_raw_data[dst_start..dst_start + self.layer_bytes].copy_from_slice(
                &self.raw_data[src_start..src_start + self.layer_bytes]
            );
        }
        
        let qtensor = candle_core::quantized::ggml_file::qtensor_from_ggml(
            self.ggml_dtype,
            &temp_raw_data,
            vec![flat_indices.len(), self.hidden_dim],
            device,
        )?;
        
        let dequantized = qtensor.dequantize(device)?;
        let reshaped = dequantized.reshape((batch_size, seq_len, self.hidden_dim))?;
        Ok(reshaped)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        rotary_dim: usize,
        rope_frequency: f32,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        // Exponents are scaled using head_dim rather than rotary_dim in Proportional RoPE (p-RoPE)
        let theta: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos, rotary_dim })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?.contiguous()?;

        if self.rotary_dim < head_dim {
            // Split q and k along the last dimension (the embedding dimension head_dim)
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
            let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?.contiguous()?;

            // Apply RoPE rotation only to the partial dimension slice
            let q_rot_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            let k_rot_embed = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;

            // Concatenate back along the last dimension
            let q_embed = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?.contiguous()?;
            let k_embed = Tensor::cat(&[&k_rot_embed, &k_pass], D::Minus1)?.contiguous()?;
            Ok((q_embed, k_embed))
        } else {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            Ok((q_embed, k_embed))
        }
    }

    fn apply_rotary_emb_q(
        &self,
        q: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (_b_sz, _h, seq_len, head_dim) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?.contiguous()?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?.contiguous()?;

        if self.rotary_dim < head_dim {
            let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
            let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?.contiguous()?;
            let q_rot_embed = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
            let q_embed = Tensor::cat(&[&q_rot_embed, &q_pass], D::Minus1)?.contiguous()?;
            Ok(q_embed)
        } else {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            Ok(q_embed)
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,

    // Optional normalization for Q and K (Gemma 2 does not use them)
    attention_q_norm: Option<RmsNorm>,
    attention_k_norm: Option<RmsNorm>,

    inp_gate: Option<QMatMul>,
    post_norm: Option<RmsNorm>,

    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,

    mlp: Mlp,

    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,

    sliding_window_size: Option<usize>,

    rotary_embedding: RotaryEmbedding,
    neg_inf: Tensor,

    // Per-layer token embeddings (PLE) for Gemma 4 E2B
    per_layer_embeddings: Option<CustomEmbedding>,
    per_layer_embeddings_proj: Option<QMatMul>,

    // Per-layer output scaling factor (layer_scalar)
    layer_output_scale: Option<Tensor>,
}

impl LayerWeights {
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(sliding_window_size) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sliding_window_size < i {
                            0u32
                        } else {
                            1u32
                        }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        let mask = if index_pos > 0 {
            let mask0 = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_sz, 1, seq_len, seq_len + index_pos))?
            .to_dtype(dtype)
    }

    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: Option<(Tensor, Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply Q/K normalizations conditionally if they are defined
        let q = match &self.attention_q_norm {
            Some(norm) => norm.forward(&q.contiguous()?)?,
            None => q,
        };
        let k = match &self.attention_k_norm {
            Some(norm) => norm.forward(&k.contiguous()?)?,
            None => k,
        };

        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[&k_cache, &k], 2)?;
                    let v = Tensor::cat(&[&v_cache, &v], 2)?.contiguous()?;
                    (k, v)
                }
            }
        };
        let new_kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Restore Attention Soft-Capping (prevent winner-take-all softmax saturation)
        let mut attn_weights = ((attn_weights / 50.0)?.tanh()? * 50.0)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, self.q_dim))?;

        Ok((self.attention_wo.forward(&attn_output)?, new_kv_cache))
    }

}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: CustomEmbedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    shared_kv_layers: usize,
    kv_caches: Vec<Option<(Tensor, Tensor)>>,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let prefix = ["gemma4", "gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma4");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let _head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let _head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let shared_kv_layers = ct.metadata
            .get(&format!("{prefix}.attention.shared_kv_layers"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(0);
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_pattern: Vec<bool> = md_get("attention.sliding_window_pattern")
            .and_then(|v| {
                if let candle_core::quantized::gguf_file::Value::Array(arr) = v {
                    let mut b = Vec::new();
                    for item in arr {
                        if let candle_core::quantized::gguf_file::Value::Bool(val) = item {
                            b.push(*val);
                        } else if let candle_core::quantized::gguf_file::Value::I32(val) = item {
                            b.push(*val != 0);
                        }
                    }
                    Ok(b)
                } else {
                    candle_core::bail!("sliding_window_pattern is not an array")
                }
            })
            .unwrap_or_default();

        let _sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .or_else(|_| md_get("rope.freq_base_swa"))
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let _rope_freq_scaling_factor = md_get("rope.scaling.factor")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR);

        let max_seq_len = ct.metadata
            .get(&format!("{prefix}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(8192)
            .min(8192);

        let rope_dim = md_get("rope.dimension")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .or_else(|_| md_get("rope.dimension_count").and_then(|m| Ok(m.to_u32()? as usize)))
            .unwrap_or(key_length);

        let rope_dim_sliding = md_get("rope.dimension_count_swa")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(rope_dim / 2);

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_emb_info = ct.tensor_infos.get("token_embd.weight").ok_or_else(|| {
            candle_core::Error::Msg("cannot find token_embd.weight in GGUF metadata".to_string())
        })?;
        let tok_elems = tok_emb_info.shape.elem_count();
        let tok_block_size = tok_emb_info.ggml_dtype.block_size();
        let tok_type_size = tok_emb_info.ggml_dtype.type_size();
        let tok_size_bytes = tok_elems / tok_block_size * tok_type_size;
        
        let mut tok_raw_data = vec![0u8; tok_size_bytes];
        reader.seek(std::io::SeekFrom::Start(ct.tensor_data_offset + tok_emb_info.offset))?;
        reader.read_exact(&mut tok_raw_data)?;
        
        let tok_shape = tok_emb_info.shape.dims();
        let vocab_size = tok_shape[0];
        let tok_hidden_dim = tok_shape[1];
        
        let tok_embeddings = CustomEmbedding::new(
            tok_raw_data,
            tok_emb_info.ggml_dtype,
            vocab_size,
            tok_hidden_dim,
            None,
            tok_hidden_dim,
        );

        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };

        let ple_len = md_get("embedding_length_per_layer_input")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(256);

        // Load the per_layer_token_embd.weight GGUF tensor raw bytes once as shared Arc
        // to avoid allocating separate huge buffers and prevent startup lag.
        let ple_shared_data = if let Some(tensor_info) = ct.tensor_infos.get("per_layer_token_embd.weight") {
            let tensor_elems = tensor_info.shape.elem_count();
            let block_size = tensor_info.ggml_dtype.block_size();
            let type_size = tensor_info.ggml_dtype.type_size();
            let size_in_bytes = tensor_elems / block_size * type_size;
            
            let mut raw_data = vec![0u8; size_in_bytes];
            reader.seek(std::io::SeekFrom::Start(ct.tensor_data_offset + tensor_info.offset))?;
            reader.read_exact(&mut raw_data)?;
            
            let original_shape = tensor_info.shape.dims();
            let vocab_size = original_shape[0]; // 262144
            let total_cols = original_shape[1]; // 8960
            
            Some((std::sync::Arc::<[u8]>::from(raw_data), tensor_info.ggml_dtype, vocab_size, total_cols))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mut layer_head_dim = 256;
            let attention_q_norm = match ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device) {
                Ok(t) => {
                    let dims = t.shape().dims();
                    layer_head_dim = if dims.len() > 1 { dims[1] } else { dims[0] };
                    Some(RmsNorm::from_qtensor(t, rms_norm_eps)?)
                }
                Err(_) => None,
            };

            let attention_k_norm = match ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device) {
                Ok(t) => Some(RmsNorm::from_qtensor(t, rms_norm_eps)?),
                Err(_) => None,
            };

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;

            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            let q_dim = attention_wq.shape().dims()[0];
            let k_dim = attention_wk.shape().dims()[0];
            let layer_n_head = q_dim / layer_head_dim;
            let layer_n_kv_head = k_dim / layer_head_dim;
            let is_sliding = sliding_window_pattern.get(layer_idx).copied().unwrap_or(true);

            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let layer_rope_dim = if is_sliding {
                rope_dim_sliding
            } else {
                rope_dim
            };

            let sliding_window_size = is_sliding.then_some(sliding_window_size);

            let inp_gate = match ct.tensor(reader, &format!("{prefix}.inp_gate.weight"), device) {
                Ok(t) => Some(QMatMul::from_qtensor(t)?),
                Err(_) => None,
            };
            let post_norm = match ct.tensor(reader, &format!("{prefix}.post_norm.weight"), device) {
                Ok(t) => Some(RmsNorm::from_qtensor(t, rms_norm_eps)?),
                Err(_) => None,
            };

            let kv_source_layer = if layer_idx >= (block_count - shared_kv_layers) {
                layer_idx - (block_count - shared_kv_layers)
            } else {
                layer_idx
            };

            let per_layer_embeddings = match ct.tensor_infos.get(&format!("{prefix}.token_embd_per_layer.weight")) {
                Some(tensor_info) => {
                    let mut raw_data = vec![0u8; tensor_info.shape.elem_count() / tensor_info.ggml_dtype.block_size() * tensor_info.ggml_dtype.type_size()];
                    reader.seek(std::io::SeekFrom::Start(ct.tensor_data_offset + tensor_info.offset))?;
                    reader.read_exact(&mut raw_data)?;
                    let dims = tensor_info.shape.dims();
                    Some(CustomEmbedding::new(
                        raw_data,
                        tensor_info.ggml_dtype,
                        dims[0],
                        dims[1],
                        None,
                        dims[1],
                    ))
                }
                None => {
                    if let Some((ref shared_data, dtype, vocab_size, total_cols)) = ple_shared_data {
                        let mut custom_emb = CustomEmbedding::new(
                            vec![],
                            dtype,
                            vocab_size,
                            ple_len,
                            Some(layer_idx),
                            total_cols,
                        );
                        custom_emb.raw_data = shared_data.clone();
                        Some(custom_emb)
                    } else {
                        None
                    }
                }
            };

            let per_layer_embeddings_proj = match ct.tensor(reader, &format!("{prefix}.proj.weight"), device) {
                Ok(t) => Some(QMatMul::from_qtensor(t)?),
                Err(_) => None,
            };

            let rotary_embedding = RotaryEmbedding::new(
                layer_head_dim,
                layer_rope_dim,
                layer_rope_frequency,
                max_seq_len,
                device,
            )?;

            let layer_output_scale = match ct.tensor(reader, &format!("{prefix}.layer_output_scale.weight"), device) {
                Ok(qt) => Some(qt.dequantize(device)?),
                Err(_) => None,
            };

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                inp_gate,
                post_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: layer_n_head,
                n_kv_head: layer_n_kv_head,
                head_dim: layer_head_dim,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                per_layer_embeddings,
                per_layer_embeddings_proj,
                layer_output_scale,
            })
        }

        Ok(Self {
            tok_embeddings,
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            shared_kv_layers,
            kv_caches: vec![None; block_count],
        })
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;

        let mut layer_in = self.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        for (_layer_idx, layer) in self.layers.iter().enumerate() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(_b_sz, seq_len, index_pos, x.dtype(), x.device())?)
            };

            // Inject PLE residual signal if configured for this layer (Gemma 4 E2B)
            let mut current_layer_in = layer_in.clone();

            if let Some(ple) = &layer.per_layer_embeddings {
                let mut ple_emb = ple.forward(x)?;
                
                // The inp_gate projects current_layer_in (1536) down to 256 to gate the PLE signal
                if let Some(gate) = &layer.inp_gate {
                    let gate_val = candle_nn::ops::sigmoid(&gate.forward(&current_layer_in)?)?;
                    ple_emb = (ple_emb * gate_val)?;
                }

                let ple_signal = if let Some(proj) = &layer.per_layer_embeddings_proj {
                    proj.forward(&ple_emb)?
                } else {
                    ple_emb
                };
                current_layer_in = (&current_layer_in + &ple_signal)?;
            }

            let residual = &current_layer_in;
            let x = layer.attention_norm.forward(&current_layer_in)?;
            
            let kv_source_idx = if _layer_idx >= (self.layers.len() - self.shared_kv_layers) {
                _layer_idx - (self.layers.len() - self.shared_kv_layers)
            } else {
                _layer_idx
            };
            
            let cache = self.kv_caches[kv_source_idx].clone();
            let (x, new_cache) = layer.forward_attn(&x, attention_mask.as_ref(), index_pos, cache)?;
            if _layer_idx == kv_source_idx {
                self.kv_caches[kv_source_idx] = new_cache;
            }

            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;

            let x = match &layer.post_norm {
                Some(norm) => norm.forward(&x)?,
                None => x,
            };

            let x = match &layer.layer_output_scale {
                Some(scale) => x.broadcast_mul(scale)?,
                None => x,
            };

            let x_sum = x.sum_all()?.to_scalar::<f32>()?;
            if x_sum.is_nan() || x_sum.is_infinite() {
                println!("🚨🚨🚨 WARNING: Layer {} Signal Sum exploded to NaN/Inf: {} 🚨🚨🚨", _layer_idx, x_sum);
            }

            layer_in = x;
        }

        let x = layer_in.i((.., seq_len - 1, ..))?.contiguous()?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;
        let output = ( (output / 30.0)?.tanh()? * 30.0 )?;

        Ok(output)
    }
}
