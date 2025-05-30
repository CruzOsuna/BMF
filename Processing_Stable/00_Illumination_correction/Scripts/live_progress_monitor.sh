#!/bin/bash

# Script: live_progress_monitor.sh

# Color Configurations
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
MAGENTA='\e[35m'
CYAN='\e[36m'
RESET='\e[0m'
BOLD='\e[1m'

# Spinner characters
SPINNER=('|' '/' '-' '\\')

# Paths
INPUT_HOST_DIR="/media/cruz/Mice/t-CycIF_mice_p53/1_Registration/RCPNLS/"
OUTPUT_HOST_DIR="/media/cruz/Mice/t-CycIF_mice_p53/00_Illumination correction/Output"
FILE_TYPE="rcpnl"

# State Variables
declare -a SPEED_HISTORY
START_TIME=$(date +%s)
LAST_UPDATE=$START_TIME
INITIAL_COUNT=0
SPINNER_INDEX=0

# Function to Draw Progress
draw_progress() {
    local current=$1
    local total=$2
    local elapsed=$3
    local speed=$4
    local remaining=$5
    
    local cols=$(tput cols)
    local bar_width=$((cols - 45))  # Adjusted for new layout
    local percent=$((current * 100 / total))
    local filled=$((current * bar_width / total))
    local empty=$((bar_width - filled))
    
    # Progress Bar
    printf "\r${BOLD}${CYAN}["
    for ((i=0; i<filled; i++)); do
        if [ $i -lt $((bar_width / 3)) ]; then printf "${RED}█"; 
        elif [ $i -lt $((2 * bar_width / 3)) ]; then printf "${YELLOW}█"; 
        else printf "${GREEN}█"; fi
    done
    printf "${RESET}%${empty}s" | tr ' ' '.'
    printf "]"

    # Progress Stats
    printf " ${BOLD}${MAGENTA}%3d%%${RESET}" "$percent"
    printf " ${GREEN}%d/${total}${RESET}" "$current"
    
    # Spinner Animation
    printf " ${CYAN}${SPINNER[$SPINNER_INDEX]}${RESET}"
    SPINNER_INDEX=$(( (SPINNER_INDEX + 1) % 4 ))

    printf "\n${BOLD}${CYAN}Time Elapsed:${RESET} ${BLUE}$(date -u -d @$elapsed +%H:%M:%S)${RESET}"
    printf " ${GREEN}Speed: %d files/min${RESET}" "$speed"
    if [ $speed -gt 0 ]; then
        if [ $remaining -gt 86400 ]; then
            printf " ${RED}ETA: Calculating...${RESET}"
        else
            printf " ${RED}ETA: %s${RESET}" "$(date -u -d @$remaining +%H:%M:%S)"
        fi
    fi

    tput cuu1
}

# Speed Calculation (Using Exponential Moving Average)
calculate_speed() {
    local current=$1
    local last=$2
    local delta_time=$3

    [ $delta_time -eq 0 ] && delta_time=1  # Avoid division by zero
    local instant_speed=$(( (current - last) * 60 / delta_time ))

    # Exponential Moving Average (EMA) for smoother speed estimation
    local alpha=0.3
    if [ ${#SPEED_HISTORY[@]} -eq 0 ]; then
        SPEED_HISTORY=($instant_speed)
    else
        local last_speed=${SPEED_HISTORY[-1]}
        SPEED_HISTORY+=($(awk "BEGIN {print int(($alpha * $instant_speed) + ((1 - $alpha) * $last_speed))}"))
    fi

    echo ${SPEED_HISTORY[-1]}
}

# Get Progress Function
get_progress() {
    local total_input=$(find "$INPUT_HOST_DIR" -type f -name "*.$FILE_TYPE" | wc -l)
    local completed_pairs=$(find "$OUTPUT_HOST_DIR" -type f \( -name "*-ffp.tif" -o -name "*-dfp.tif" \) -exec basename {} \; | 
                          sed 's/-ffp.tif//;s/-dfp.tif//' | sort | uniq | wc -l)
    echo "$completed_pairs $total_input"
}

# Main
echo -e "${BOLD}${CYAN}=== Illumination Correction Progress Monitor ===${RESET}"
echo -e "${BLUE}Input: ${YELLOW}$INPUT_HOST_DIR${RESET}"
echo -e "${BLUE}Output: ${YELLOW}$OUTPUT_HOST_DIR${RESET}"
echo -e "${BOLD}${RED}Press Ctrl+C to exit${RESET}\n"

read INITIAL TOTAL < <(get_progress)
LAST_COUNT=$INITIAL

while true; do
    read CURRENT TOTAL < <(get_progress)
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    DELTA_TIME=$((CURRENT_TIME - LAST_UPDATE))
    DELTA_COUNT=$((CURRENT - LAST_COUNT))

    # Speed Calculation
    if [ $DELTA_COUNT -gt 0 ]; then
        SPEED=$(calculate_speed $CURRENT $LAST_COUNT $DELTA_TIME)
        REMAINING=$(( (TOTAL - CURRENT) * 60 / ($SPEED + 1) ))  # +1 to prevent division by zero
    else
        SPEED=0
        REMAINING=0
    fi

    # Update Every 5s or if Changes Detected
    if [ $DELTA_COUNT -gt 0 ] || [ $((CURRENT_TIME - LAST_UPDATE)) -ge 5 ]; then
        draw_progress "$CURRENT" "$TOTAL" "$ELAPSED" "$SPEED" "$REMAINING"
        LAST_UPDATE=$CURRENT_TIME
        LAST_COUNT=$CURRENT
    fi

    # Check Completion
    if [ $CURRENT -ge $TOTAL ] && [ $TOTAL -gt 0 ]; then
        echo -e "\n\n${BOLD}${GREEN}✔ Process Completed!${RESET}"
        echo -e "${BOLD}Total Time: ${YELLOW}$(date -u -d @$ELAPSED +%H:%M:%S)${RESET}"
        exit 0
    fi

    sleep 1
done
