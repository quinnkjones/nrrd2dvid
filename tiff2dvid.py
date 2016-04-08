# -*- coding: utf-8 -*-

"""TIFF2DVID.
Written by Michael Morehead.
Based heavily on NRRD2DVID by Quinn Jones.
https://github.com/quinnkjones/nrrd2dvid
"""
from libdvid import DVIDNodeService, DVIDServerService, DVIDException
import argparse
import json
import numpy as np
import pdb
import matplotlib as plt


parser = argparse.ArgumentParser(description="Batch mode nrrd file to dvid migration script")
existingNode = parser.add_argument_group('existing node', 'for working with a node that already exists on the dvid server')
parser.add_argument('address', metavar='address', help='address to a valid dvid server in the form x.x.x.x:yyyy')
parser.add_argument('file', metavar='tiff_file', help='filepath for uploading to dvid')
existingNode.add_argument('--uuid', '-u', metavar='uuid', help='minimal uid of the node to access on the dvid server')
newNode = parser.add_argument_group('new node', 'for creating a new node before migrating the nrrd files')
newNode.add_argument('--alias', '-a', metavar='alias', help='alias for a new node to create')
newNode.add_argument('--description', '-d', metavar='description', help='description for new node')
newNode.add_argument('--segmentation', '-s', action='store_true', help='flags data as a segmentation block in the case that there was no indication in the header')
args = parser.parse_args()
addr = args.address


if args.alias:
    service = DVIDServerService(addr)
    uid = service.create_new_repo(args.alias, args.description)
else:
    uid = args.uuid


def push_to_dvid(method, handle, data, preoffset=(0, 0, 0), throttle=False, compress=True, chunkDepth=512):
    """Function for pushing to DVID."""
    zsize = data.shape[0]
    numsplits = zsize / 512
    offset = 0
    pdb.set_trace()
    for i in xrange(numsplits):
        seg = data[i * chunkDepth:(i + 1) * chunkDepth, :, :]
        offsetTuple = (preoffset[0] + i * chunkDepth, preoffset[1], preoffset[2])

        method(handle, seg, offsetTuple, throttle, compress)

    offset = numsplits * chunkDepth
    seg = data[offset:zsize, :, :]
    offsetTuple = (preoffset[0] + offset, preoffset[1], preoffset[2])
    method(handle, seg, offsetTuple, throttle, compress)


def yieldtoDvid(method, handle, header, filehandle, dtype, compress=True):
    """Generator for posting to DVID."""
    pdb.set_trace()
    for col, row, z, data in nrrd.iterate_data(header, input_tiff, handle):
        data = np.ascontiguousarray(data)
        data = data.astype(dtype)
        res = method(handle, data, (z, row, col), False, compress)
        print res


with open(args.file, "rb") as input_tiff:
    header = args.file
    data = plt.imread(tiff_file)

    service = DVIDNodeService(addr, uid)
    kvname = 'headers'
    if service.create_keyvalue(kvname):
        service.put(kvname, args.file, headerJson)
    else:
        service.put(kvname, args.file, headerJson)
        # we should check if the key is there and warn the user to avoid overwriting when not desired

    # data = np.ascontiguousarray(nrrd.read_data(header, input_tiff, args.file))

    reshaper = []

    for dim in data.shape:
        if dim % 32 != 0:
            newmax = (dim / 32 + 1) * 32
        else:
            newmax = dim
        reshaper += [(0, newmax - dim)]

    data = np.pad(data, reshaper, mode='constant')

    d2 = data.copy()
    data = None
    pdb.set_trace()
    if args.segmentation or header['keyvaluepairs'].get('seg', '') == 'true':
        d2 = d2.astype(np.uint64)
        try:
            service.create_labelblk(args.file)
        except DVIDException:
            print 'warning override data?'

        push_to_dvid(service.put_labels3D, args.file, d2)
        # yieldtoDvid(service.put_labels3D, args.file, header, input_tiff, np.uint64)
    else:
        if header['keyvaluepairs'].get('seg', None) is None:
            print 'warning header value for seg is not set nor is flag'
        d2 = d2.astype(np.uint8)
        try:
            service.create_grayscale8(args.file)
        except DVIDException:
            print "warnging override data"
        push_to_dvid(service.put_gray3D, args.file, d2, compress=False)
        # yieldtoDvid(service.put_gray3D, args.file, header, input_tiff, np.uint8, compress=False)
