# File naming
- MatWAND accepts either [.mat or binary files](/Inputs.md). 
- The files can be separated [in MatWAND](#matwand-separation), or manually by the user [before](#user-separation) MatWAND analysis starts.

---

### File naming template
The file names should contain:

1) **userString** (e.g. animal5).

2) **ID** or unique identification number (e.g. 101).

3) **condition** (e.g. baseline).

** Template = **userString_ID_condition** ** -> e.g. animal5\_101_baseline.

- ID needs to be between underscores. 
- This is essential for matching the files based on animal or subject.
- For example, in **animal1\_101_baseline.mat** where the identifier is ***101***, will be matched with  **animal1\_102_drug.mat**.

### Naming Rules
1. The three elements have to be separated by an underscore (ID needs to be between underscores)

      * userString_ID_condition **(e.g. animal5\_101_baseline)** &nbsp;:heavy_check_mark: 
      
      * userString-ID-condition **(e.g. animal5-101-baseline)**  &nbsp;&nbsp;&nbsp; :x: 
     
      * userString^condition^ID **(e.g. animal5^101^baseline)** &nbsp; :x: 

2. The order of the three elements (user-string, ID, condition) must remain unchanged

      * userString_ID_condition **(e.g. animal56\_105_drug)** &nbsp; :heavy_check_mark: 
      
      * condition_userString_ID **(e.g. drug\_animal56_105)** &nbsp; :x: 
      
      * userString_condition_ID **(e.g. animal56\_drug_105)** &nbsp; :x:  

3. Underscores should not be used withing elements

      * userString_ID_condition **(e.g. PFCanimal4\_101_baseline)** &nbsp; :heavy_check_mark:
      
      * user_String_ID_condition **(e.g. PFC\_animal4**_101_wt_baseline.mat) :x:
      
      * userString_ID_cond_ition (e.g. PFCanimal4\_101_**base_line)** &nbsp;  :x:
      
4. The ID should only consist of integers

      * 101 :heavy_check_mark:, 12b :x:, zab :x:, 5 :heavy_check_mark:, 55*12 :x:, 55_12 :x:

5. Optional: More than one condition can be appended to the end of the file name

      * userString_ID_condition1_condition2 **(e.g. animal4\_101_wt_baseline)**
      
      * userString_ID_condition1_condition2 **(e.g. animal4\_101_ko_baseline)**
---

### MatWAND separation

- MatWAND will prompt the user for file separation based on comments in [.mat file](/Inputs.md).
- Useful for short files where the comments and comment times have been recorded.

Consider an example where we want to separate the following files: 

      animal55_1_wt.mat, animal65_2_wt.mat, animal72_3_.mat, , animal81_4_wt.mat
      
with the following comments:
      
      baseline, drugX, wash
      
- MatWAND will split the original names based on the comments and comment times as can be seen below. 

| Original Name | Condition 1 | Condition 2 | Condition 3 |
| ------------- | -------- | ----- | ---- |
| animal55_1_wt.mat | animal55_1_wt_baseline.mat | animal55_1_wt_drugX.mat | animal55_1_wt_wash.mat |
| animal65_2_wt.mat | animal65_2_wt_baseline.mat | animal65_2_wt_drugX.mat | animal65_2_wt_wash.mat |
| animal72_3_wt.mat | animal72_3_wt_baseline.mat | animal72_3_wt_drugX.mat | animal72_3_wt_wash.mat |
| animal81_4_wt.mat | animal81_4_wt_baseline.mat | animal81_4_wt_drugX.mat | animal81_4_wt_wash.mat |
       
---

### User separation

- Files can be also be separated by the user before initiating MatWAND analysis.
- Useful for long files that need to be anyway saved separately.
- The files should be separated based on their conditions with a matching ID as can be seen below.

| Condition 1 | Condition 2 | Condition 3 |
| -------- | ----- | ---- |
| animal55_1_baseline.bin | animal55_1_drugX.bin | animal55_1_wash.bin |
| animal65_2_baseline.bin | animal65_2_drugX.bin | animal65_2_wash.bin |
| animal72_3_baseline.bin | animal72_3_drugX.bin | animal72_3_wash.bin |
| animal81_4_baseline.bin | animal81_4_drugX.bin | animal81_4_wash.bin |

---

**[<< Back to Main Page](/index.md)**
