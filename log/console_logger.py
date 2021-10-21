import datetime


class console_logger:
	def log_msg(self, msg):
		line = f"{datetime.datetime.now()}: {msg}"
		print(line)
