use std::{env, fs, path::Path};
use tera::{Tera, Context};
use walkdir::WalkDir;

// Adapted from: https://codecrash.me/an-opengl-preprocessor-for-rust
fn generate_shaders() {
    const SHADER_SRC_PATH: &str = "src/shaders";

    let tera = Tera::new(format!("{}/**/*", SHADER_SRC_PATH).as_str()).unwrap();
    println!("cargo:rerun-if-changed={}/", SHADER_SRC_PATH);

    let mut context = Context::new();

    //context.insert("module_data", &MODULE_DATA);

    let output_path = Path::new(env::var("OUT_DIR").unwrap().as_str())
                            .join("shaders");
    fs::create_dir_all(&output_path).unwrap();

    
    WalkDir::new("src/shaders")
                .into_iter()
                .filter_map(Result::ok)
                .filter(|e| !e.file_type().is_dir())
                .for_each(|entry| {
        
        let src_file_path = entry.path()
                         .strip_prefix(SHADER_SRC_PATH)
                         .unwrap();
        let src_file_path_str = src_file_path.to_string_lossy()
                                        .replace("\\", "/");
        let output_file_path = output_path.join(src_file_path);
        println!("{}", src_file_path_str);

        let result = tera.render(&src_file_path_str, &context).unwrap();
        fs::create_dir_all(output_file_path.parent().unwrap()).unwrap();
        fs::write(output_file_path, result).unwrap();
        println!("cargo:rerun-if-changed={}/{}", SHADER_SRC_PATH, src_file_path_str);
    });
}

fn main() {
    //generate_shaders()
}