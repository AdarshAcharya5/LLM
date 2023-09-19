import os
class Dataloader:

    def __init__(self, input_file : str, output_file : None, source_directory=None):
        self.source_directory = source_directory
        self.input_file = input_file
        self.output_file = output_file
        assert self.input_file is not None, "Input file not specified"

    # Extracts file content from a single file
    def extract(self, input_file=None, replace_char=' '):
        if input_file is None:
            input_file = self.input_file
        with open(input_file, "r", encoding="utf-8") as current_file:
            file_content = current_file.read()
        file_content = self.remove_invalid_chars(file_content, replace_char)
        current_file.close()
        return file_content

    # Returns merged file content from all files in a directory
    def merge(self, output_file=None, delimiter="\n", replace_char=' '):
        if output_file is None:
            output_file = self.output_file
        assert self.source_directory is not None and self.output_file is not None,\
            "Source directory or Output file not specified"

        with open(output_file, "a", encoding="utf-8") as merged_file:
            for filename in os.listdir(self.source_directory):
                assert filename.endswith(".txt"), "Only .txt files are supported"
                if filename.endswith(".txt"):
                    file_path = os.path.join(self.source_directory, filename)
                    with open(file_path, "r", encoding="utf-8") as current_file:
                        file_content = current_file.read()
                    file_content = self.remove_invalid_chars(file_content, replace_char)
                    merged_file.write(file_content)
                    merged_file.write(delimiter)
        return self.extract(output_file)

    # Removes invalid characters from file content
    def remove_invalid_chars(self, file_content : str, replace_char=' '):
        lines = file_content.split("\n")
        for i in range(len(lines)):
            for ch in lines[i]:
                if ord(ch) < 32 or ord(ch) > 126:
                    lines[i] = lines[i].replace(ch, replace_char)
        return '\n'.join(lines)


