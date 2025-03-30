use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{ Path, PathBuf };
use walkdir::WalkDir;
use rayon::prelude::*;
use std::sync::{ Arc, Mutex };

fn process_dataset(
    input_labels_path: &Path,
    input_img_path: &Path,
    output_path: &Path
) -> Result<(), Box<dyn std::error::Error>> {
    // Create output directories
    fs::create_dir_all(output_path.join("images"))?;
    fs::create_dir_all(output_path.join("labels"))?;

    let imgs_dir = output_path.join("images");
    let labels_dir = output_path.join("labels");

    // Load JSON data
    let file_data = fs::read_to_string(input_labels_path)?;
    let data: Vec<Value> = serde_json::from_str(&file_data)?;

    let missing_files = Arc::new(Mutex::new(Vec::new()));
    data.par_iter().for_each(|d| {
        let filename = d["name"].as_str().unwrap();
        let mut src_file: Option<PathBuf> = None;

        for entry in WalkDir::new(input_img_path).into_iter().filter_map(Result::ok) {
            if entry.file_type().is_file() && entry.file_name() == filename {
                src_file = Some(entry.path().to_path_buf());
                break;
            }
        }

        if let Some(src) = src_file {
            // Copy image to imgs directory
            if let Err(e) = fs::copy(&src, imgs_dir.join(filename)) {
                eprintln!("Error copying {}: {}", filename, e);
                return;
            }

            // Write temporary JSON file
            let base_name = Path::new(filename).file_stem().unwrap().to_str().unwrap();
            let json_dst = output_path.join(format!("{}.json", base_name));
            if let Err(e) = fs::write(&json_dst, serde_json::to_string(d).unwrap()) {
                eprintln!("Error writing JSON for {}: {}", filename, e);
            }
        } else {
            let mut missing = missing_files.lock().unwrap();
            missing.push(filename.to_string());
        }
    });

    let missing = missing_files.lock().unwrap();
    for file in missing.iter() {
        println!("Missing: {:?}", input_img_path.join(file));
    }

    // Filter labels: keep only good labels
    let good_labels = ["car", "person", "rider", "bus", "truck", "bike"];

    let json_entries: Vec<_> = fs
        ::read_dir(output_path)?
        .filter_map(Result::ok)
        .filter(
            |entry|
                entry
                    .path()
                    .extension()
                    .and_then(|s| s.to_str()) == Some("json")
        )
        .collect();

    json_entries.par_iter().for_each(|entry| {
        match fs::read_to_string(entry.path()) {
            Ok(content) => {
                if let Ok(mut json_data) = serde_json::from_str::<Value>(&content) {
                    if
                        let Some(labels) = json_data
                            .get_mut("labels")
                            .and_then(|v| v.as_array_mut())
                    {
                        labels.retain(|l| {
                            l.get("category")
                                .and_then(|cat| cat.as_str())
                                .map_or(false, |cat| good_labels.contains(&cat))
                        });
                    }
                    if
                        let Err(e) = fs::write(
                            entry.path(),
                            serde_json::to_string_pretty(&json_data).unwrap()
                        )
                    {
                        eprintln!("Error writing filtered JSON: {}", e);
                    }
                }
            }
            Err(e) => eprintln!("Error reading JSON file {:?}: {}", entry.path(), e),
        }
    });

    // Convert JSON labels to YOLO text files
    let num_to_class = [
        ("car", 0),
        ("person", 1),
        ("rider", 2),
        ("bus", 3),
        ("truck", 4),
        ("bike", 5),
    ]
        .iter()
        .cloned()
        .collect::<std::collections::HashMap<_, _>>();

    let json_files: Vec<_> = fs
        ::read_dir(output_path)?
        .filter_map(Result::ok)
        .filter(
            |entry|
                entry
                    .path()
                    .extension()
                    .and_then(|s| s.to_str()) == Some("json")
        )
        .collect();

    json_files.par_iter().for_each(|entry| {
        match fs::read_to_string(entry.path()) {
            Ok(content) => {
                if let Ok(json_data) = serde_json::from_str::<Value>(&content) {
                    let mut output_lines = Vec::new();

                    if let Some(labels) = json_data.get("labels").and_then(|v| v.as_array()) {
                        for label in labels {
                            let category = label["category"].as_str().unwrap();
                            let class_num = num_to_class.get(category).unwrap();
                            let box2d = &label["box2d"];
                            let x1 = box2d["x1"].as_f64().unwrap();
                            let y1 = box2d["y1"].as_f64().unwrap();
                            let x2 = box2d["x2"].as_f64().unwrap();
                            let y2 = box2d["y2"].as_f64().unwrap();

                            let image_width: f64 = 1280.0;
                            let image_height: f64 = 720.0;

                            let width = (x2 - x1) / image_width;
                            let height = (y2 - y1) / image_height;
                            let middle_x = (x2 + x1) / (2.0 * image_width);
                            let middle_y = (y2 + y1) / (2.0 * image_height);

                            output_lines.push(
                                format!(
                                    "{} {} {} {} {}",
                                    class_num,
                                    middle_x,
                                    middle_y,
                                    width,
                                    height
                                )
                            );
                        }
                    }

                    // Get the file stem (filename without extension)
                    let file_stem = entry.path().file_stem().unwrap().to_owned();
                    // Create path in the labels directory
                    let txt_filename = labels_dir.join(file_stem).with_extension("txt");

                    if let Ok(mut txt_file) = fs::File::create(&txt_filename) {
                        if let Err(e) = writeln!(txt_file, "{}", output_lines.join("\n")) {
                            eprintln!("Error writing YOLO file: {}", e);
                        }
                    }
                    // Remove temporary JSON file
                    if let Err(e) = fs::remove_file(entry.path()) {
                        eprintln!("Error removing temp JSON file: {}", e);
                    }
                }
            }
            Err(e) => eprintln!("Error reading JSON file for conversion: {}", e),
        }
    });

    println!("Processed dataset at {:?}", output_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Process validation data
    let val_label_path = Path::new(
        "../raw_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
    );
    let val_img_path = Path::new("../raw_data/bdd100k/images/100k/val");
    let val_output_path = Path::new("../formatted_data/val");
    process_dataset(val_label_path, val_img_path, val_output_path)?;

    // Process training data
    let train_label_path = Path::new(
        "../raw_data/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
    );
    let train_img_path = Path::new("../raw_data/bdd100k/images/100k/train");
    let train_output_path = Path::new("../formatted_data/train");
    process_dataset(train_label_path, train_img_path, train_output_path)?;

    Ok(())
}
