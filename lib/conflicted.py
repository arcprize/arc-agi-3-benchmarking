library(conflicted)
library(dplyr)

filter(mtcars, cyl == 8)
#> Error:
#> ! [conflicted] filter found in 2 packages.
#> Either pick the one you want with `::`:
#> • dplyr::filter
#> • stats::filter
#> Or declare a preference with `conflicts_prefer()`:
#> • `conflicts_prefer(dplyr::filter)`
#> • `conflicts_prefer(stats::filter)`
