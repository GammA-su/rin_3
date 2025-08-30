suite:
	python3 guardian_agi_min.py --memdir .guardian_mem --suite quick --strict
acl:
	python3 guardian_agi_min.py --task acl --memdir .guardian_mem --strict
ae:
	python3 guardian_agi_min.py --task ae --strict
nsv:
	python3 guardian_agi_min.py --task nsv
oe:
	python3 guardian_agi_min.py --task oe
