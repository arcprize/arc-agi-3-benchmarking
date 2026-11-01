# modules expects you to namespace all package functions
dplyr <- modules::import_package('dplyr')
dplyr$filter(mtcars, cyl == 8)

# import expects you to explicit load functions
import::from(dplyr, select, arrange, dplyr_filter = filter)
dplyr_filter(mtcars, cyl == 8)
