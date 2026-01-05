from src.main.utils.jssp_instance import JSSPInstance, load_orlib_jobshop

jssp_instance: JSSPInstance = load_orlib_jobshop("datasets/ft06.txt")

print(jssp_instance.n_jobs)        # 6
print(jssp_instance.n_machines)    # 6
print(jssp_instance.total_operations)  # 36

print(jssp_instance.jobs)