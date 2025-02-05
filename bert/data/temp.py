def interleave_files(file_a, file_b, file_c):
    # Read lines from File A
    with open(file_a, "r", encoding="utf-8") as fa:
        lines_a = fa.readlines()
    
    # Read lines from File B
    with open(file_b, "r", encoding="utf-8") as fb:
        lines_b = fb.readlines()

    # Interleave A and B
    interleaved_data = []
    len_a, len_b = len(lines_a), len(lines_b)
    max_len = max(len_a, len_b)

    for i in range(max_len):
        if i < len_a:
            interleaved_data.append(lines_a[i])  # Take from A
        if i < len_b:
            interleaved_data.append(lines_b[i])  # Take from B

    # Write to File C
    with open(file_c, "w", encoding="utf-8") as fc:
        fc.writelines(interleaved_data)

    print(f"Interleaving complete: {len(interleaved_data)} entries written to {file_c}")

# Example Usage
file_a = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/data/retain/retain_test.jsonl"
file_b = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/data/forget/forget_test.jsonl"
file_c = "/media/jagadeesh/New Volume/Jagadeesh/GS-LoRA/bert/data/retain/test.jsonl"

interleave_files(file_a, file_b, file_c)
