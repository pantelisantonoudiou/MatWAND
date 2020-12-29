# File preparation
- MatWAND accepts either [.mat or binary files](/Docs/Inputs.md). 
- The files can be separated in MatWAND, or manually by the user before MatWAND analysis starts.

---

## Naming
- MatWAND requires the files to contain a unique number identifier between underscores.
- This is essential for matching the files based on animal or subject. 
- For example, in **animal1\_101_baseline.mat** where the identifier is ***101***.
- This will be matched with  **animal1\_102_drug.mat**.
- **Underscores in the file name should only be used for identifiers and for adding more than one conditions.**

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



---

**[<< Back to Main Page](/README.md)**
