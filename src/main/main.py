from src.main.utils.jssp_instance import JSSPInstance, load_orlib_jobshop

jssp_instance: JSSPInstance = load_orlib_jobshop("datasets/ft06.txt")

print(jssp_instance.n_jobs)
print(jssp_instance.n_machines)
print(jssp_instance.total_operations)
print(jssp_instance.jobs)