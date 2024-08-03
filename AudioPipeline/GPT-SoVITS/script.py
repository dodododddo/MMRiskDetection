import os

def rename_files_in_directory(directory, new_first_letter):
    for filename in os.listdir(directory):
        if filename[0].isalpha():
            new_name = new_first_letter + filename[1:]
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

directory_path = '/data1/home/jrchen/MMRiskDetection/AudioPipeline/GPT-SoVITS/tests'
new_first_letter = 'X'  
rename_files_in_directory(directory_path, new_first_letter)
