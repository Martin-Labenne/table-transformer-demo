import os
import tempfile
import subprocess

TABLE_TRANSFORMER_FOLDER_NAME = 'tableTransformer'

def get_table_transformer_repo(): 
    subprocess.run(['git', 'clone', 'git@github.com:microsoft/table-transformer.git', TABLE_TRANSFORMER_FOLDER_NAME])
    os.chdir(f'./{TABLE_TRANSFORMER_FOLDER_NAME}')
    subprocess.run(['git', 'reset', '--hard', '16d124f616109746b7785f03085100f1f6247575'])
    os.chdir('./..')

def create_environment(file_path='./environment.yml'): 
    subprocess.run(['conda', 'env', 'create', '-f', file_path])

def update_environment(file_path='./environment.yml'):
    subprocess.run(['conda', 'env', 'update', '-f', file_path])

if __name__ == "__main__":
    get_table_transformer_repo()

    with open(f'./{TABLE_TRANSFORMER_FOLDER_NAME}/environment.yml', 'r') as env_file: 
        first_line = env_file.readline()
        first_line_key_value = first_line.split(':')
        first_line_key_value[1] = ' table-transformer\n'

        first_line = ':'.join(first_line_key_value)
        other_lines = env_file.readlines()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_env_file:
            temp_env_file.write(first_line)
            temp_env_file.writelines(other_lines)
            
            temp_env_file_path = temp_env_file.name
      
    create_environment(file_path = temp_env_file_path)
    update_environment()


