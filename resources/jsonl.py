import json

def read_jsonl(file_path):
    """Reads a JSONL file line by line and yields dictionary objects."""
    print(f"--- Reading data from: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON on line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")

def write_jsonl(data_list, output_path):
    """Writes a list of dictionaries to a JSONL file."""
    print(f"--- Writing data to: {output_path} ---")
    with open(output_path, 'w', encoding='utf-8') as file:
        for entry in data_list:
            # ensure_ascii=False keeps non-English characters readable
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')
    print("Write complete!")

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # 1. Define dummy framework data to write
    sample_framework_data = [
        {"id": 1, "component": "Navbar", "version": 2.3, "status": "stable"},
        {"id": 2, "component": "Sidebar", "version": 2.3, "status": "deprecated"},
        {"id": 3, "component": "Footer", "version": 2.4, "status": "experimental"}
    ]
    
    # Define file names
    input_filename = "MyFramework.jsonl"
    output_filename = "Filtered_Framework.jsonl"
    
    # 2. Write the sample data to a file
    write_jsonl(sample_framework_data, input_filename)
    
    # 3. Read and filter data dynamically (Generator memory-efficient way)
    filtered_data = []
    for item in read_jsonl(input_filename):
        # Example condition: Filter for version 2.3 items only
        if item.get("version") == 2.3:
            filtered_data.append(item)
            
    # 4. Save the filtered results to a new file
    write_jsonl(filtered_data, output_filename)
