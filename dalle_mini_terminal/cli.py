#!/usr/bin/env python3

import re

def main(arguments):
	config=dict()
	positional=[]
	pattern=re.compile(r"(?:-(?:d|h|x|o|V|v)|--(?:dalle|help|output|version|vqgan))(?:=.*)?$")
	consuming,needing,wanting=None,0,0
	attached_value=None
	def log(*values): pass

	if "--debug-gap-behavior" in arguments:
		def log(*values): print(*values)
	while len(arguments) and arguments[0]!="--":
		if arguments[0]=="--debug-gap-behavior":
			arguments.pop(0)
			continue
		log(f'processing {arguments[0]}...')
		if consuming is not None:
			log(f'option {consuming} is consuming')
			if config[consuming] is None:
				config[consuming]=arguments.pop(0)
				log(f'option {consuming} = {config[consuming]}')
			else:
				config[consuming].append(arguments.pop(0))
				log(f'option {consuming} = {config[consuming]}')
			needing-=1
			wanting-=1
			if wanting==0:
				log(f'option {consuming} is no longer consuming')
				consuming,needing,wanting=None,0,0
		elif pattern.match(arguments[0]):
			log(f'{arguments[0]} matched an option')
			option = arguments.pop(0).lstrip('-')
			if '=' in option:
				log(f'{option} has an attached value')
				option,attached_value=option.split('=',1)
			log(f'{option} is an option')
			if option=="dalle":
				if attached_value is not None:
					config["dalle"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["dalle"]=None
					consuming,needing,wanting="dalle",1,1
			elif option=="help":
				if attached_value is not None:
					message=(
						'unexpected value while parsing "help"'
						' (expected 0 values)'
					)
					raise ValueError(message) from None
				config["help"]=True
			elif option=="output":
				if attached_value is not None:
					config["output"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["output"]=None
					consuming,needing,wanting="output",1,1
			elif option=="version":
				if attached_value is not None:
					message=(
						'unexpected value while parsing "version"'
						' (expected 0 values)'
					)
					raise ValueError(message) from None
				config["version"]=True
			elif option=="vqgan":
				if attached_value is not None:
					config["vqgan"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["vqgan"]=None
					consuming,needing,wanting="vqgan",1,1
			elif option=="d":
				if attached_value is not None:
					config["dalle"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["dalle"]=None
					consuming,needing,wanting="dalle",1,1
			elif option=="h":
				if attached_value is not None:
					message=(
						'unexpected value while parsing "help"'
						' (expected 0 values)'
					)
					raise ValueError(message) from None
				config["help"]=True
			elif option=="x":
				if attached_value is not None:
					message=(
						'unexpected value while parsing "help"'
						' (expected 0 values)'
					)
					raise ValueError(message) from None
				config["help"]=True
			elif option=="o":
				if attached_value is not None:
					config["output"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["output"]=None
					consuming,needing,wanting="output",1,1
			elif option=="V":
				if attached_value is not None:
					message=(
						'unexpected value while parsing "version"'
						' (expected 0 values)'
					)
					raise ValueError(message) from None
				config["version"]=True
			elif option=="v":
				if attached_value is not None:
					config["vqgan"]=attached_value
					attached_value=None
					consuming,needing,wanting=None,0,0
				else:
					config["vqgan"]=None
					consuming,needing,wanting="vqgan",1,1
		else:
			positional.append(arguments.pop(0))
	if needing>0:
		message=(
			f'unexpected end while parsing "{consuming}"'
			f' (expected {needing} values)'
		)
		raise ValueError(message) from None
	for argument in arguments[1:]:
		if argument=="--debug-gap-behavior":
			continue
		positional.append(argument)
	return config,positional

if __name__=="__main__":
	import sys
	cfg,pos = main(sys.argv[1:])
	cfg = {k:v for k,v in cfg.items() if v is not None}
	if len(cfg):
		print("Options:")
		for k,v in cfg.items():
			print(f"{k:20} = {v}")
	if len(pos):
		print("Positional arguments:", ", ".join(pos))
