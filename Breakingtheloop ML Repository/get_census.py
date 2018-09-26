
import json
import requests
import urllib3
import census
from urllib.parse import urlparse
from census import Census
from us import states
from ast import literal_eval

codes = {'tot':'B01003_001E','pov:':'B17005_002E', 'coll':'B23006_023E', 'NotMarrUnempMale':'B12006_006E','NotMarrUnempFemale':'B12006_011E', 'MarrUnempMale':'B12006_017E',
               'MarrUnempFemale':'B12006_022E','SepUnempMale':'B12006_029E', 'SepUnempFemale':'B12006_033E',  'DivMale':'B12006_050E',  'DivFemale':'B12006_055E' , 'WidMale':'B12006_039E', 'WidFemale':'B12006_044E'}

# codes from: https://api.census.gov/data/2015/acs/acs5/variables.html

key ='1c775b832899a80d4481940aff6a6c27ccaa9623'
c = Census(key)

## A. Using zipcode

def get_educzip(zipc):
    '''
    Using zipcode (5 digit) 
    '''
    if type(zipc) != str:
        zipc = str(int(float(zipc)))
    if len(zipc) == 5:
        res = c.acs5.zipcode('B01003_001E,B23006_023E', zipc, year=2011)
        
        if res:
            tot = int(res[0]['B01003_001E'])
            coll = int(res[0]['B23006_023E'])
            if tot !=0:

                return coll / tot

    else:
        return None
    
def get_unempz(zipc):
    if type(zipc) != str:
        zipc = str(int(zipc))
        if len(zipc) == 5:
            tot, nmarm, nmarf, marm, marf, sm, sf, dm, df, wm, wf = codes['tot'], \
            codes['NotMarrUnempMale'], codes['NotMarrUnempFemale'], codes['MarrUnempMale'], \
                                             codes['MarrUnempFemale'],codes['SepUnempMale'], \
                                             codes['SepUnempFemale'], codes['DivMale'], \
                                             codes['DivFemale'], codes['WidMale'], codes['WidFemale']

            try:
                res = c.acs5.zipcode('B01003_001E,B12006_006E', zipc,  year=2011)
                res1 = c.acs5.zipcode('B12006_011E,B12006_017E', zipc, year=2011)
                res2 = c.acs5.zipcode('B12006_022E,B12006_029E', zipc, year=2011)
                res3 = c.acs5.zipcode('B12006_033E,B12006_050E', zipc, year=2011)
                res4 = c.acs5.zipcode('B12006_055E,B12006_039E', zipc,  year=2011)
                res5 = c.acs5.zipcode('B12006_044E', zipc, year=2011)
                totpop  = int(res[0][tot])
                if totpop != 0:
                    unemp = int(res[0][nmarm]) + int(res1[0][nmarf]) + int(res1[0][marm]) + int(res2[0][marf]) + int(res2[0][sm]) + int(res3[0][sf]) + int(res3[0][dm]) + int(res4[0][df]) + int(res4[0][wm]) + int(res5[0][wf])
                    return unemp / totpop
            except Exception as ex:
                pass
    else:
        return None
    
def get_povz(zipc):
    '''
    Using zipcode (5 digit) 
    '''
    if type(zipc) != str:
        zipc = str(int(zipc))
        if len(zipc) == 5:
            res = c.acs5.zipcode('B01003_001E,B17005_002E', zipc, year=2011) 
            if res:
                tot = int(res[0]['B01003_001E'])
                coll = int(res[0]['B17005_002E'])
                if tot != 0:
                    return coll / tot
            
    else:
        return None
    

## 2. Using tract

def get_unemp(fips):
    if fips > 0:
        fips = str(int(fips))
        state, county, tract = fips[:2], fips[2:5], fips[5:]
        tot, nmarm, nmarf, marm, marf, sm, sf, dm, df, wm, wf = codes['tot'], \
        codes['NotMarrUnempMale'], codes['NotMarrUnempFemale'], codes['MarrUnempMale'], \
                                         codes['MarrUnempFemale'],codes['SepUnempMale'], \
                                         codes['SepUnempFemale'], codes['DivMale'], \
                                         codes['DivFemale'], codes['WidMale'], codes['WidFemale']
                    
        varstring = tot+','+nmarm+','+nmarf+','+marm+','+marf+','+sm+','+sf
        res = c.acs5.state_county_tract('B01003_001E,B12006_006E', state, county, tract,  year=2010)
        res1 = c.acs5.state_county_tract('B12006_011E,B12006_017E', state, county, tract,  year=2010)
        res2 = c.acs5.state_county_tract('B12006_022E,B12006_029E', state, county, tract,  year=2010)
        res3 = c.acs5.state_county_tract('B12006_033E,B12006_050E', state, county, tract,  year=2010)
        res4 = c.acs5.state_county_tract('B12006_055E,B12006_039E', state, county, tract,  year=2010)
        res5 = c.acs5.state_county_tract('B12006_044E', state, county, tract,  year=2010)
        
        totpop  = res[0][tot]
        if totpop != 0:
            unemp = res[0][nmarm] + res1[0][nmarf] + res1[0][marm] + res2[0][marf] + res2[0][sm] + res3[0][sf] + res3[0][dm] + res4[0][df] + res4[0][wm] + res5[0][wf]
            return unemp / totpop
    else:
        return None
    
def get_educ(fips):
    if fips > 0:
        fips = str(int(fips))
        state = fips[:2]
        
        county = fips[2:5]
        tract = fips[5:]
        res = c.acs5.state_county_tract('B01003_001E,B23006_023E', state, county, tract, year=2010)
        tot = res[0]['B01003_001E']
        if tot != 0:
            coll = res[0]['B23006_023E']
            return coll / tot
    else:
        return None
        
        

def get_pov(fips):
    if fips > 0:
        fips = str(int(fips))
        state = fips[:2]
        county = fips[2:5]
        tract = fips[5:]
        res = c.acs5.state_county_tract('B01003_001E,B17005_002E', state, county, tract,  year=2010)
        tot = res[0]['B01003_001E']
        if tot != 0:
            pov = res[0]['B17005_002E']
            return pov / tot
    else:
        return None
    

        

