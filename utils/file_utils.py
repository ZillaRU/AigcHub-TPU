import os
import stat

def add_executable_permission(file_path):
    """
    为指定文件添加执行权限。

    参数:
    file_path (str): 要修改权限的文件的路径。
    """
    try:
        # 获取当前文件的权限
        current_permissions = os.stat(file_path).st_mode

        # 添加执行权限，对于所有者、组和其他用户
        new_permissions = current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

        # 应用新的权限
        os.chmod(file_path, new_permissions)
        print(f"Excecution permission of {file_path} is added.")
    except Exception as e:
        print(f"Fail to modify the access permission of {file_path}, Error:{e}")