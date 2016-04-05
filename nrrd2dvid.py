import nrrd
from libdvid import DVIDNodeService, DVIDServerService
import argparse
import json

parser = argparse.ArgumentParser(description="Batch mode nrrd file to dvid migration script")
existingNode = parser.add_argument_group('existing node','for working with a node that already exists on the dvid server')
parser.add_argument('address',metavar='address', help='address to a valid dvid server in the form x.x.x.x:yyyy')
existingNode.add_argument('uuid',metavar='uuid',help='minimal uid of the node to access on the dvid server')
newNode = parser.add_argument_group('new node','for creating a new node before migrating the nrrd files')
newNode.add_argument('--alias','-a', metavar='alias',help='alias for a new node to create')
newNode.add_argument('--description','-d',metavar='description',help='description for new node')
parser.add_argument('file',metavar='nrrdFile',help='filepath for uploading to dvid')
args = parser.parse_args()
addr = args.address

if args.alias:
    service = DVIDServerService(addr)
    uid = service.create_new_repo(args.alias,args.description)
else:
    uid = args.uuid

with open(args.nrrdfile,"rb") as inputnrrd:
    headerJson = json.dumps(nrrd.read_header(inputnrrd))

service = DVIDNodeService(addr,uuid)
kvname = 'headers'
if service.create_keyvalue(kvname):
    service.put(kvname,args.nrrdfile,headerJson)

#verification
print service.get(kvname
