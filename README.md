# rl-opt 

![batch](https://img.shields.io/github/downloads/idsc-frazzoli/rl-opt/total.svg)

Library for relaxed lexicographic optimizaton (lexicographic semiorders) in Python

The library was developed with the following objectives in mind
* investigate the dependency of the EBO decision and the optimal solutions given a lexicographic semiorder,
* examine how the different parameters affect the size of the candidate set and the total number of comparisons made and 
* create visualization to help understand lexicographic semiorders as preference structures.

## Lexicographic Semiorder

A lexicographic semiorder compares elements of a Cartesian Product of several sets lexicographically. A element x is said to be smaller than
another element y if there exists an index where x_i is better than y_i by a threshold sigma.


## Visualizations

See Chapters 4.5 and 6.2 in *Multi-objective Optimization using Preference Structures* for the explanation of the visualizations and the outline of the simulations.

### Optimal solutions

<table>
<tr>
<td>

![heatmapbest-1](https://user-images.githubusercontent.com/44157083/61792470-1c160300-ae1d-11e9-9184-5723a5d534df.png)

Minimum optimal set size vs. threshold size (two objectives)

<td>

![heatmapaverage-1](https://user-images.githubusercontent.com/44157083/61792299-baee2f80-ae1c-11e9-9d15-47a3e1174cb4.png)

Average optimal set size vs. threshold size (two objectives)

<td>

![heatmapworst-1](https://user-images.githubusercontent.com/44157083/61792545-4d8ece80-ae1d-11e9-9a28-ded533f12dbf.png)

Maximum optimal set size vs. threshold size (two objectives)

</tr>
</table>


<table>
<tr>
<td>

![ratio2-1](https://user-images.githubusercontent.com/44157083/61794020-972ce880-ae20-11e9-858c-9f64c89a7380.png)

Frequnency of <a href="https://www.codecogs.com/eqnedit.php?latex=\text{Min}^\circ(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Min}^\circ(X)" title="\text{Min}^\circ(X)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\min(X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min(X)" title="\min(X)" /></a> being non-empty.

<td>

![ratio1-1](https://user-images.githubusercontent.com/44157083/61794290-2c2fe180-ae21-11e9-8c5a-e5de51f47b03.png)

Optimal set size vs. threshold size (three objectives)

</tr>
</table>
  
### EBO procedure

<table>
<tr>
<td>

![animation](https://user-images.githubusercontent.com/44157083/61469058-d40b6200-a97e-11e9-8e5b-d353fab65fc6.gif)

2D EBO procedure

<td>

![animation](https://user-images.githubusercontent.com/44157083/61469173-04eb9700-a97f-11e9-8db0-36e11d2c8704.gif)

3D EBO procedure

</tr>
</table>

<table>
<tr>
<td>

![cand_thresholdsize-1](https://user-images.githubusercontent.com/44157083/61794343-48cc1980-ae21-11e9-9dc4-5db43073920d.png)

Candidate set size vs. threshold size (3 objectives)

<td>

![comp_thresholdsize-1](https://user-images.githubusercontent.com/44157083/61794492-9ba5d100-ae21-11e9-98a2-5154f2953f0d.png)

Total number of comparisons vs. threshold size (3 objectives)

</tr>
</table>

<table>
<tr>
<td>

![cand_samplesize-1](https://user-images.githubusercontent.com/44157083/61794568-bed08080-ae21-11e9-8c3c-b899eefb1550.png)

Candidate set size vs. sample size (3 objectives)

<td>
  
![comp_samplesize-1](https://user-images.githubusercontent.com/44157083/61794652-e6bfe400-ae21-11e9-8d74-691ce86e8be8.png)

Total number of comparisons vs. sample size (3 objectives)

</tr>
</table>


## Contributors

André Stoll

## References

* *Multi-objective Optimization using Preference Structures*
by André Stoll

---
