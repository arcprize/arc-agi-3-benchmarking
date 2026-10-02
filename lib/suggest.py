library(dplyr)
conflicts_prefer(dplyr::filter)
conflict_scout()
#> 2 conflicts
#> • `filter()`: dplyr
#> • `lag()`: dplyr and stats

#> [conflicted] Will prefer dplyr::filter over any other package.
filter(mtcars, am & cyl == 8)
#>                 mpg cyl disp  hp drat   wt qsec vs am gear carb
#> Ford Pantera L 15.8   8  351 264 4.22 3.17 14.5  0  1    5    4
#> Maserati Bora  15.0   8  301 335 3.54 3.57 14.6  0  1    5    8
