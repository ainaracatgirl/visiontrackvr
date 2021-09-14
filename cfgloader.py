class ConfigKeys:
    def __init__(self,kvpairs):self.values=kvpairs
    def bool(self, key):return self.values[key].lower()=='true'
    def int(self, key):return int(self.values[key])
    def float(self, key):return float(self.values[key])
    def str(self, key):return self.values[key][1:-1]
def load(stream,close=True):
    data=stream.read()
    if close:stream.close()
    kvpairs={};lines=data.splitlines()
    for line in lines:
        if line.strip().startswith("#"):continue
        if line.strip()=="":continue
        parts=line.strip().split("=");key=parts[0].strip();value=parts[1].strip();kvpairs[key]=value
    return ConfigKeys(kvpairs)
def dump(stream,config_keys,top_comment=None):
    if top_comment is not None:
        stream.write("# "+str(top_comment)+"\n")
    keys=list(config_keys.values.keys())
    for key in keys:
        stream.write(str(key)+"="+str(config_keys.values[key])+"\n")
    stream.close()