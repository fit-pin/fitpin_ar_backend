import subprocess

# Step 1: Conda 패키지 내보내기
subprocess.run("conda env export -f environment.yml --from-history", shell=True)

# Step 2: pipdeptree 사용하여 최상위 패키지 목록 추출
pipdeptree_output = subprocess.check_output("pipdeptree --freeze", shell=True).decode('utf-8')

# Step 3: 최상위 패키지 필터링
top_level_packages = []
for line in pipdeptree_output.splitlines():
    if '==' in line and line[0].isalpha():
        top_level_packages.append(line)

# Step 4: Environment 파일 UTF-16LE로 수정
with open("environment.yml", "a", encoding='utf-8') as env_file:
    env_file.write("  - pip:\n")
    for package in top_level_packages:
        env_file.write(f"    - {package}\n")

print("environment.yml 파일이 업데이트되었습니다.")