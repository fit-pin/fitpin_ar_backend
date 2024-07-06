import subprocess

#environment.yml 생성
result = subprocess.run("conda env export -f environment.yml --no-builds", shell=True)
print("environment.yml 파일생성")

pipdeptree_output = subprocess.check_output("pipdeptree --freeze", shell=True).decode('utf-8')

top_level_packages = []
for line in pipdeptree_output.splitlines():
    if '==' in line and line[0].isalpha():
        top_level_packages.append(line)
        
#requirements.txt 생성
with open("requirements.txt", "w", encoding='utf-8') as env_file:
    for package in top_level_packages:
        env_file.write(f"{package}\n")

print("requirements.txt 파일생성")