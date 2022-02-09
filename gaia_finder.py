#!/usr/bin/env python3

import astropy.units as u
import numpy as np
import argparse as ap

from astropy.io import fits
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from tqdm.contrib import tenumerate
from astropy import wcs

Vizier.ROW_LIMIT = -1

parser = ap.ArgumentParser(description='Make GAIA image', formatter_class=ap.ArgumentDefaultsHelpFormatter)
parser.add_argument("target", type=str, help="Object Identifier")
parser.add_argument("--fov", type=float, default=1., help="Finder FOV (arcmin)")
parser.add_argument("--zeromag", type=float, default=20., help="Magnitude zero point")
parser.add_argument("--maglim", type=float, default=18., help="Magnitude cutoff for GAIA catalog")
parser.add_argument("--pxscl", type=float, default=20., help="Pixel scale (mas)")
parser.add_argument("--fwhm", type=float, default=0.06, help="FWHM (arcsec)")
parser.add_argument("--save", type=str, default="finder", help="Output name")
parser.add_argument("--cfa", action='store_true', help="Use cfa mirror instead of strasburg.")
args = parser.parse_args()

TARGET=args.target
FOV=args.fov*60
ZEROMAG=args.zeromag
PXSCL=args.pxscl/1000.
FWHM=args.fwhm
MAGLIM=args.maglim
SAVE=args.save
if args.cfa:
    Vizier.SERVER = "http://vizier.cfa.harvard.edu/"
    Simbad.SIMBAD_URL = "http://simbad.cfa.harvard.edu/simbad/sim-script"

def gaussian(p, x, y):
    return p[0]*np.exp(-((x-p[1])**2+(y-p[2])**2)*0.5/p[3]**2)/(2*np.pi)

def get_gaia(target, fov):

    result = Vizier.query_region(target, fov=fov*u.arcsec, catalog="I/350/gaiaedr3", column_filters={'Gmag': '<%.f' % MAGLIM})
    return result[0][("RA_ICRS", "DE_ICRS", "Gmag")]

def do_2mass(target, fov):

    result = Vizier.query_region(target, fov=fov*u.arcsec, catalog="II/246")
    f = open("%s_2MASS.reg" % SAVE, "w")
    f.write("# Region file format: DS9 version 4.1\n" \
            "global color=magenta dashlist=8 3 fov=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n" \
            "icrs\n")
    for target in result[0]:
        f.write("circle(%.7f,%.7f,0.6\")\n" % (target["RAJ2000"], target["DEJ2000"]))
    
    f.close()

def get_coo(target):

    coo = Simbad.query_object(target)
    coo["RA"].info.format = "%s"
    coo["DEC"].info.format = "%s"
    coo = SkyCoord("%s %s" % (coo[0][1], coo[0][2]), unit=(u.hourangle, u.deg))

    return {"RA" : coo.ra.deg, "DEC" : coo.dec.deg}

def do_image(target, fov, pxscl, fwhm, zeromag):

    table = get_gaia(target, fov)
    coo = get_coo(target)
    
    f = open("%s_GAIA.reg" % SAVE, "w")
    f.write("""# Region file format: DS9 version 4.1
global color=green dashlist=8 3 fov=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
icrs
""")

    sz = np.int64(FOV/PXSCL)
    im = np.zeros([sz, sz])

    X = np.arange(im.shape[0])-im.shape[0]//2
    XX, YY = np.meshgrid(X, X)

    for _, target in tenumerate(table):
        
        ra, dec = -(target["RA_ICRS"]-coo["RA"])*3600/PXSCL*np.cos(target["DE_ICRS"]/180.*np.pi), (target["DE_ICRS"]-coo["DEC"])*3600/PXSCL
        cnts = 10**((ZEROMAG-target["Gmag"])/2.5)

        f.write("circle(%.7f,%.7f,0.2\")\n" % (target["RA_ICRS"], target["DE_ICRS"]))
  
        im[:,:] += gaussian([cnts, ra, dec, FWHM/PXSCL], XX, YY)

    print(im.shape)

    f.close()

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [im.shape[0]/2.+1, im.shape[0]/2.+1]
    w.wcs.cdelt = np.array([-PXSCL/3600., PXSCL/3600.])
    w.wcs.crval = [coo["RA"], coo["DEC"]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    fits.PrimaryHDU(im, header=w.to_header()).writeto("%s_img.fits" % SAVE, overwrite=True)

if __name__=="__main__":
    do_image(TARGET, FOV, PXSCL, FWHM, ZEROMAG)
    do_2mass(TARGET, FOV)
