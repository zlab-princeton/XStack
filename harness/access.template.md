# TPU access

A TPU machine is allocated to this task for its duration. It must stay up
(rules at the end).

- Google Cloud @ACCEL@: one host, @NUM_CHIPS@ TPU chips (single host, no
  multi-host setup needed). TPU name `@TPU_NAME@`, zone `@TPU_ZONE@`.
- Ubuntu 22.04, remote user `@TPU_USER@`, stock Python 3.10, no ML stack
  preinstalled, ~100 GB local disk (large datasets and repeated checkpoints
  can exhaust it)
- Address: @TPU_IP@ (reachable from this workspace's machine with the key
  `~/.ssh/id_ed25519`, which opens this machine only)

Run a command on it:

    ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=30 @TPU_USER@@@TPU_IP@ 'command'

Copy files to it:

    scp -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -r local_path @TPU_USER@@@TPU_IP@:remote_path

Connections can time out intermittently; set generous connect timeouts and
retry. There is no cloud CLI access from this workspace: the machine is
reached only by ssh and scp with the key above.

Notes:

- The machine has internet access; installing packages there works.
- A dropped ssh session kills its foreground child: run long remote commands
  under `nohup` (or `tmux`) and poll their logs.
- Your final submission is collected from THIS workspace, then run on that
  machine.

Rules (the machine is shared infrastructure and must stay up):

- Never reboot, shut down or power off the machine, and never stop or
  restart system services. If the TPU looks stuck, a process of yours is
  still holding the devices: kill your own processes and wait for them to
  exit.
- Keep your files under the `@TPU_USER@` home directory, /tmp and /dev/shm,
  and only touch processes you started.
