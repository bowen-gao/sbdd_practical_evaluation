.PHONY: default

1:
	tmux has-session -t 1 2>/dev/null || tmux new-session -d -s 1
	tmux attach-session -t 1

0:
	tmux has-session -t 0 2>/dev/null || tmux new-session -d -s 0
	tmux attach-session -t 0