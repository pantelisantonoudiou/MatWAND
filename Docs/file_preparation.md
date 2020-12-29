# File preparation
- MatWAND accepts either [.mat or binary files](/Docs/Inputs.md). 
- The files can be separated in MatWAND, or manually by the user before MatWAND analysis starts.

---

### File naming template
The file names should contain:

1) **userString**, e.g. animal5.
2) unique identifier number (**ID**), e.g. 101.
3) **condition**, e.g. baseline.

** Template = **userString_ID_condition** ** -> e.g. animal5\_101_baseline.

### General naming Rules
- The three elements have to be separated by an underscore.

userString_ID_condition (e.g. animal5\_101_baseline) :heavy_check_mark: 

userString-ID-condition (e.g. animal5-101-baseline) :x: 

userString^condition^ID (e.g. animal5^101^baseline) :x: 

- The order of the three elements (user-string, ID, condition) must remain unchanged.
:x: condition_user_string_ID, :x: user_string_condition_ID

- ID needs to be between underscores.
- This is essential for matching the files based on animal or subject.
- The identifier should only consist of integers (101 :heavy_check_mark:, a2b :x:).
- For example, in **animal1\_101_baseline.mat** where the identifier is ***101***.
- This will be matched with  **animal1\_102_drug.mat**.

***Underscores in the file name should only be used for identifiers and for adding more than one conditions.**
***Underscores in the file name should only be used for identifiers and for adding more than one conditions.**



:x: **animal\_1_101_wt_baseline.mat**

:x: **PFC\_animal1_101_wt_baseline.mat**

:heavy_check_mark: **animal1\_101_wt_baseline.mat**

:heavy_check_mark: **animal1\_101_baseline.mat**

---

### 1) MatWAND separation
- MatWAND will prompt the user for file separation based on comments in [.mat file](/Docs/Inputs.md).

Consider an example where we want to separate the following files: 

      animal55_1_wt.mat, animal65_2_wt.mat, animal72_3_.mat, , animal81_4_wt.mat
      
and comments:
      
      baseline, drugX, wash
      
- MatWAND will split the original names based on the comments as can be seen below. 

| Original Name | baseline | drugX | wash |
| ------------- | -------- | ----- | ---- |
| animal55_1_wt.mat | animal55_1_wt_baseline.mat | animal55_1_wt_drugX.mat | animal55_1_wt_wash.mat |
| animal65_2_wt.mat | animal65_2_wt_baseline.mat | animal65_2_wt_drugX.mat | animal65_2_wt_wash.mat |
| animal72_3_wt.mat | animal72_3_wt_baseline.mat | animal72_3_wt_drugX.mat | animal72_3_wt_wash.mat |
| animal81_4_wt.mat | animal81_4_wt_baseline.mat | animal81_4_wt_drugX.mat | animal81_4_wt_wash.mat |
       
---

### 2) User separation before MatWAND initiation

- Files can be also be separated by user before initiating MatWAND analysis.
- The files should be separated based on their conditions.
- Each animal must have a matching

---

**[<< Back to Main Page](/README.md)**
