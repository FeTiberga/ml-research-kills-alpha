import json
import paramiko  # pip install paramiko

with open("vm_config.json") as f:
    config = json.load(f)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

if "ssh_key_path" in config:
    ssh.connect(
        config["host"],
        username=config["username"],
        key_filename=config["ssh_key_path"],
        port=config["port"]
    )
else:
    ssh.connect(
        config["host"],
        username=config["username"],
        password=config["password"],
        port=config["port"]
    )

stdin, stdout, stderr = ssh.exec_command("echo Hello from VM")
print(stdout.read().decode())
ssh.close()
