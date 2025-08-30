suite:      ## Run quick suite in strict mode with memory
	python3 guardian_agi_min.py --memdir .guardian_mem --suite quick --strict
acl:        ## Promote a new concept (edit CONCEPT=)
	python3 guardian_agi_min.py --task acl --memdir .guardian_mem --concept "$(CONCEPT)"
ae:         ## A/B adopt knobs if better
	python3 guardian_agi_min.py --task ae --strict
nsv:        ## Run NSV batch proof
	python3 guardian_agi_min.py --task nsv
oe:         ## Log one episode
	python3 guardian_agi_min.py --task oe
