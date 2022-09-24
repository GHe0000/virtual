import json
import yaml
test_file = open(test_yaml_file,"r")

generate_dict = yaml.load(test_file,Loader=yaml.FullLoader)
generate_json = json.dumps(generate_dict,sort_keys=False,indent=4,separators=(',',': '))