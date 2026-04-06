# 6h_R2beta_spatialbins.R
#
# Computes semi-partial R² (R²β) for each fixed effect in:
#   DV ~ people_bin * area_label + (1 | participant)
# for all DVs in the bin-level dataset used in Python.
#
# Output:
#   C:/LocoGaze/data/group/mixedmodel_R2beta_spatialbins.csv
#
# Requires: lme4, r2glmm, dplyr, readr

# ---------------- PACKAGES ----------------
# install.packages(c("lme4", "r2glmm", "dplyr", "readr"))  # already done
library(lme4)
library(r2glmm)
library(dplyr)
library(readr)

# ---------------- PATHS ----------------
base_dir  <- "C:/LocoGaze/data"
group_dir <- file.path(base_dir, "group")
input_csv <- file.path(group_dir, "mixedmodel_spatialbins.csv")
out_csv   <- file.path(group_dir, "mixedmodel_R2beta_spatialbins.csv")

# ---------------- LOAD DATA ----------------
df <- read_csv(input_csv, show_col_types = FALSE)

# Ensure factors are properly coded
df <- df %>%
  mutate(
    participant = factor(participant),
    people_bin  = factor(people_bin),  # "0" and "1plus"
    area_label  = factor(area_label)   # flat / cobblestone / green
  )

# ---------------- DVs TO ANALYSE ----------------
DVS <- c(
  # Gaze allocation
  "looking_people",
  "looking_floor",
  "looking_environment",
  # Gaze / depth metrics
  "radius_of_gyration",
  "spatial_entropy",
  "depth_d_s",
  "depth_head_pitch_deg",
  "depth_d_vergence",
  "depth_calib_mm",
  "gaze_y",
  "horizontal_ecc",
  # Gait metrics
  "mean_rms_LEFT",
  "stride_duration_LEFT",
  "stride_length_LEFT",
  "cadence_LEFT",
  "pace_LEFT"
)

# ---------------- FUNCTION: SAFE MODEL FIT + R2beta ----------------

get_r2beta_for_dv <- function(dv_name, data) {

  # Drop NAs on this DV
  dat <- data %>% filter(!is.na(.data[[dv_name]]))

  # At least a few participants and some variability
  if (n_distinct(dat$participant) < 3L) {
    message("Skipping ", dv_name, ": < 3 participants with data.")
    return(NULL)
  }

  # Model: DV ~ people_bin * area_label + (1 | participant)
  form <- as.formula(paste0(dv_name,
                            " ~ people_bin * area_label + (1 | participant)"))

  message("Fitting LMM for DV = ", dv_name)
  m <- try(lmer(form, data = dat, REML = FALSE), silent = TRUE)

  if (inherits(m, "try-error")) {
    message("  -> model failed for ", dv_name)
    return(NULL)
  }

  # Semi-partial R² (R²β) for fixed effects
  r2 <- try(r2beta(m, method = "sgv", partial = TRUE), silent = TRUE)
  if (inherits(r2, "try-error")) {
    message("  -> r2beta failed for ", dv_name)
    return(NULL)
  }

  r2_df <- as.data.frame(r2)
  r2_df$dv <- dv_name

  # r2beta returns columns including:
  #   Effect, Rsq, F, v1, v2, ncp, (optionally CLs depending on version)
  # To be version-robust, map them to standard names if present.
  if ("F" %in% names(r2_df) && !("F.value" %in% names(r2_df))) {
    r2_df$F.value <- r2_df$F
  }
  if ("v1" %in% names(r2_df) && !("df.num" %in% names(r2_df))) {
    r2_df$df.num <- r2_df$v1
  }
  if ("v2" %in% names(r2_df) && !("df.den" %in% names(r2_df))) {
    r2_df$df.den <- r2_df$v2
  }

  # Make sure all expected columns exist (fill with NA if not)
  for (nm in c("Rsq", "F.value", "df.num", "df.den")) {
    if (!(nm %in% names(r2_df))) {
      r2_df[[nm]] <- NA_real_
    }
  }

  # Keep a clean subset of columns
  r2_df <- r2_df %>%
    select(
      dv,
      Effect,
      Rsq,
      F.value,
      df.num,
      df.den
    )

  return(r2_df)
}

# ---------------- LOOP OVER DVs ----------------

r2_list <- lapply(DVS, get_r2beta_for_dv, data = df)
# drop NULLs
r2_list <- r2_list[!vapply(r2_list, is.null, logical(1))]

if (length(r2_list) == 0L) {
  stop("No R²β results produced for any DV.")
}

all_r2 <- bind_rows(r2_list)

# Optional: keep only fixed effects of interest (drop intercept)
all_r2 <- all_r2 %>%
  filter(Effect %in% c("people_bin", "area_label", "people_bin:area_label"))

# Rename to manuscript-friendly names
all_r2 <- all_r2 %>%
  rename(
    effect  = Effect,
    R2_beta = Rsq,
    F       = F.value,
    df_num  = df.num,
    df_den  = df.den
  )

# ---------------- SAVE RESULTS ----------------
write_csv(all_r2, out_csv)
message("Saved semi-partial R² results to: ", out_csv)
