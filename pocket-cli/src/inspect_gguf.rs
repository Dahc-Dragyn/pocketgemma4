use std::fs::File;

fn main() -> anyhow::Result<()> {
    let model_path = "H:\\pocketgemma4\\google_gemma-4-E2B-it-Q4_K_M.gguf";
    println!("Loading GGUF tensor values from: {}", model_path);
    
    let mut file = File::open(model_path)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let device = candle_core::Device::Cpu;
    
    for l in [0, 4, 33] {
        let name = format!("blk.{}.layer_output_scale.weight", l);
        match ct.tensor(&mut file, &name, &device) {
            Ok(qt) => {
                let t = qt.dequantize(&device)?;
                let val = t.to_vec1::<f32>()?;
                println!("Value for {}: {:?}", name, val);
            }
            Err(e) => {
                println!("Error loading {}: {:?}", name, e);
            }
        }
    }
    
    Ok(())
}
