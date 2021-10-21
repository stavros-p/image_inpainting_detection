import visdom
import numpy as np
import os
import datetime



class visdom_logger:
	##
	# @class visdom_logger
	# @brief Logs on Visdom server and on txt files
	def __init__(self, env_name: str, logpath: str=None, address="http://localhost", port: int=8097, batch_count: int=1, iters: int=500) -> None:
		##
		# @brief Initialize the logger
		# @param	envName		The visdom environment name
		# @param	logpath		The path to the logfile to log
		# @param	address		The address to host the visdom server
		# @param	port		The port to access the visdom server
		# @param	batchCount	If the batch size is larger than 1, visualize batchCount items from a minibatch
		super(visdom_logger, self).__init__()
		# @param	iters		Log every iters iterations
		self.env_name = env_name
		self.address = address
		self.port = port
		self.batch_count = batch_count
		self.iteration = iters
		self.instance = visdom.Visdom(server=address, port=port, env=env_name, use_incoming_socket=False)

		self.path = None
		if logpath is not None:
			if os.path.isdir(logpath) and not os.path.exists(os.path.dirname(logpath)) and logpath is not None:
				raise RuntimeError(f"The specified filepath: 'logpath' doesn't exist.")
			self.path = logpath
			fd = open(logpath, 'w')
			fd.close()


	def log_msg(self, msg: str) -> None:
		line = f"{datetime.datetime.now()}: {msg}"
		print(line)
		if self.path:
			fd = open(self.path, 'a')
			fd.write(f"{line}\n")
			fd.close()
		