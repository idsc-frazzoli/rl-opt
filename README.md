# LexSemiorderSurvivorSet
Generates 2D and 3D plots and animations of the survivor set for a given lexicographic semiorder.

## Lexicographic Semiorder

A lexicographic semiorder compares elements of a Cartesian Product of several sets lexicographically. A element x is said to be smaller than
another element y if there exists an index where x_i is better than y_i by a threshold sigma.

## Survivor Set
The survivor set is sequential generated by eliminating all elements which are not within the threshold of the minimum value of the current projection step. 
Since the data is often streamed sequentially it is necessary to retain data points which are in fact not minimal but can be in the future.

## Plots

The plots and animations display for a given input set which is streamed to the algorithm the current minimal set,
the points which have to be retained and the discarded points.


<table>
<tr>
<td>

![ anim](https://user-images.githubusercontent.com/44157083/55796422-48a24b00-5aca-11e9-81e2-5db846f78eb7.gif)

2D Survivor Set

<td>

![anim](https://user-images.githubusercontent.com/44157083/55797288-5789fd00-5acc-11e9-97ce-4abbcb898079.gif)

3D Survivor Set

</tr>
</table>

### Creating the gifs

Once the scripts have be run, the figures will be stored in a separate folder (2Dplots or 3Dplots). Using ImageMagick a gif can be created
from the figure*.png.
Go to the folder location and run (bash not python):
```
$ convert -delay 10 -loop 0 *.png anim.gif
```