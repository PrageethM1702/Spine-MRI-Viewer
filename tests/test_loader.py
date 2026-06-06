from data_loader import get_subjects_by_vendor, get_t2_path

subs = get_subjects_by_vendor("All")

print("Subjects:", len(subs))

print("Sample:", subs[0])
print("T2 path:", get_t2_path(subs[0]))