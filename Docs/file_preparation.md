# File preparation
MatWAND accepts either .mat or binary files. The files can be separated in MatWAND or before MatWAND analysis starts by the user.

---

## Naming
- MatWAND requires the files to contain a unique number identifier between underscores.
- This is essential for matching the files based on animal or subject. 
- For example, in **animal1\_101_base.mat** where the identifier is ***101***.
- This will be matched with  **animal1\_102_drug.mat**.

---

### 1) MatWAND separation
- MatWAND will prompt the user for file separation based on comments in [.mat file](/Docs/Inputs.md).

Consider an example where we want to separate the following files: 

      anima55_1_wt.mat, animal65_2_wt.mat, animal72_3_.mat, , animal81_4_wt.mat
      
and comments:
      
      baseline, drugX, wash
      
After MatWAND will result in the following files:

| Original Name | baseline | drugX | wash |
| ------------- | -------- | ----- | ---- |
| anima55_1_wt.mat | anima55_1_wt_baseline.mat | anima55_1_wt_drux.mat | anima55_1_wt_wash.mat 
| Paragraph        | Text        |
       
---

### 2) User separation

---

**[<< Back to Main Page](/README.md)**
