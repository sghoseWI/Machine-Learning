EXPLAIN
with bk_dates as (
select bk.mni_no, bk.booking_date,
case when inm.dedupe_id is null
then nix.dedupe_id
else inm.dedupe_id
end,
case when inm.dob is null
then nix.dob
else inm.dob
end
from clean.jocojimsjailbooking_hashed bk
left join clean.jocojims2inmatedata inm
on bk.mni_no = inm.mni_no 
left join clean.jocojims2nameindexdata nix 
on inm.mni_no = nix.mni_no)
SELECT distinct on (bk.dedupe_id, bk.mni_no, bk.booking_date)
bk.*, 
case when mht.lastest_mh_discharge::timestamp > bk.booking_date::timestamp
then null 
else mht.lastest_mh_discharge
end as latest
from bk_dates bk 
    left JOIN
    (
        SELECT mh.dedupe_id, mh.patid, mh.dob, MAX(mh.dschrg_date) AS lastest_mh_discharge
        FROM clean.jocomentalhealth_hashed mh
        GROUP BY mh.dedupe_id, mh.patid, mh.dob
    ) mht ON bk.dob = mht.dob and bk.dedupe_id = mht.dedupe_id
    JOIN clean.jocomentalhealth_hashed mh2 ON mht.dedupe_id = mh2.dedupe_id and mht.patid = mh2.patid AND mht.lastest_mh_discharge = mh2.dschrg_date;
