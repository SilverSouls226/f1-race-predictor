# Create local library directory
if (!dir.exists("R_libs")) dir.create("R_libs")
.libPaths(c("R_libs", .libPaths()))

# Load libraries
if (!require("tidyverse")) install.packages("tidyverse", lib="R_libs", repos="https://cloud.r-project.org")
if (!require("corrplot")) install.packages("corrplot", lib="R_libs", repos="https://cloud.r-project.org")
if (!require("ggplot2")) install.packages("ggplot2", lib="R_libs", repos="https://cloud.r-project.org")

library(tidyverse)
library(corrplot)
library(ggplot2)

# Set working directory if needed, but script assumes running from project root
# setwd("C:/Users/subbu/Desktop/Fods project")

# 1. Load Data
# We use the aggregated data for most plots to avoid overplotting, but lap-level for lap times.
df_race <- read.csv("data/processed/f1_driver_race.csv")
df_laps <- read.csv("data/processed/f1_cleaned_laps.csv") # Might take a while but needed for lap time boxplots

# Ensure plots directory exists
if (!dir.exists("plots")) {
  dir.create("plots")
}

# 2. Lap Time Distribution by Team (Boxplot)
# We need to filter for top teams to make plot readable if too many teams
top_teams <- df_race %>%
  group_by(team) %>%
  summarise(avg_points = mean(points, na.rm = TRUE)) %>%
  arrange(desc(avg_points)) %>%
  head(10) %>%
  pull(team)

df_laps_filtered <- df_laps %>% filter(team %in% top_teams)

p1 <- ggplot(df_laps_filtered, aes(x = team, y = lap_time, fill = team)) +
  geom_boxplot(outlier.shape = NA) +
  coord_cartesian(ylim = quantile(df_laps_filtered$lap_time, c(0.01, 0.99), na.rm=TRUE)) +
  theme_minimal() +
  labs(title = "Lap Time Distribution by Top 10 Teams",
       y = "Lap Time (s)", x = "Team") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/01_lap_time_distribution.png", plot = p1, width = 10, height = 6)
print("Saved plots/01_lap_time_distribution.png")

# 3. Scatter Plot: Qualifying vs Race Position
p2 <- ggplot(df_race, aes(x = grid_position, y = finish_position)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_jitter(width = 0.2, height = 0.2) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "Qualifying vs Race Position",
       x = "Grid Position", y = "Finish Position")

ggsave("plots/02_qualifying_vs_race_pos.png", plot = p2, width = 8, height = 6)
print("Saved plots/02_qualifying_vs_race_pos.png")

# 4. Correlation Heatmap
# Select numerical columns
numeric_cols <- df_race %>% select(grid_position, finish_position, points, 
                                   avg_lap_time, std_lap_time, best_lap_time, 
                                   pit_stop_count, position_gain,
                                   avg_air_temp, avg_track_temp, rain_probability) %>%
  na.omit()

cor_matrix <- cor(numeric_cols)

png("plots/03_correlation_heatmap.png", width = 800, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
dev.off()
print("Saved plots/03_correlation_heatmap.png")

# 5. Driver Skill / Performance Change over Time (Bonus)
# Let's see average finish position per season for top 5 drivers
top_drivers <- df_race %>%
  group_by(driver) %>%
  summarise(total_points = sum(points, na.rm=TRUE)) %>%
  arrange(desc(total_points)) %>%
  head(5) %>%
  pull(driver)

p3 <- df_race %>%
  filter(driver %in% top_drivers) %>%
  group_by(season, driver) %>%
  summarise(avg_pos = mean(finish_position, na.rm=TRUE)) %>%
  ggplot(aes(x = season, y = avg_pos, color = driver, group = driver)) +
  geom_line() +
  geom_point() +
  scale_y_reverse() + # Lower position is better
  theme_minimal() +
  labs(title = "Average Finish Position by Season (Top 5 Drivers)",
       y = "Average Finish Position", x = "Season")

ggsave("plots/04_driver_performance_trend.png", plot = p3, width = 10, height = 6)
print("Saved plots/04_driver_performance_trend.png")
