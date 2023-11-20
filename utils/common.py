def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')
