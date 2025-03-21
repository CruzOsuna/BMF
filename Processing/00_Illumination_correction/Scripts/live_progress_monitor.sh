#!/bin/bash

# Script: final_progress_monitor.sh
# Uso: ./final_progress_monitor.sh

# Configuración de estilo
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
MAGENTA='\e[35m'
CYAN='\e[36m'
RESET='\e[0m'
BOLD='\e[1m'

# Configuración de rutas
INPUT_HOST_DIR="/media/cruz-osuna/Mice/CycIF_mice_p53/1_Registration/RCPNLS"
OUTPUT_HOST_DIR="/media/cruz-osuna/Mice/CycIF_mice_p53/00_Illumination_correction/Output"
FILE_TYPE="rcpnl"

# Variables de estado
declare -a SPEED_HISTORY
START_TIME=$(date +%s)
LAST_UPDATE=$START_TIME
INITIAL_COUNT=0

# Función para dibujar la interfaz
draw_enhanced_progress() {
    local current=$1
    local total=$2
    local elapsed=$3
    local speed=$4
    local remaining=$5
    
    local cols=$(tput cols)
    local bar_width=$((cols - 38))
    local percent=$((current * 100 / total))
    local filled=$((current * bar_width / total))
    local empty=$((bar_width - filled))
    
    # Barra de progreso con degradado
    printf "\r${BOLD}${CYAN}Progress: ${RESET}["
    for i in $(seq 1 $filled); do
        [ $i -le $((bar_width / 3)) ] && printf "${RED}◼"
        [ $i -gt $((bar_width / 3)) ] && [ $i -le $((2 * bar_width / 3)) ] && printf "${YELLOW}◼"
        [ $i -gt $((2 * bar_width / 3)) ] && printf "${GREEN}◼"
    done
    printf "${RESET}%${empty}s" | tr ' ' '.'
    printf "]"

    # Estadísticas
    printf " ${BOLD}${MAGENTA}%3d%%${RESET}" "$percent"
    printf " ${GREEN}%d/${total}${RESET}" "$current"
    printf "\n${BOLD}${CYAN}Stats:${RESET}"
    printf " ${BLUE}⏱ %s${RESET}" "$(date -u -d @$elapsed +%H:%M:%S)"
    printf " ${GREEN}▲ %d files/min${RESET}" "$speed"
    [ $speed -gt 0 ] && printf " ${RED}⏳ %s remaining${RESET}" "$(date -u -d @$remaining +%H:%M:%S)"
    
    tput cuu1
}

# Función para cálculo de velocidad
calculate_speed() {
    local current=$1
    local last=$2
    local delta_time=$3
    
    [ $delta_time -eq 0 ] && delta_time=1  # Evitar división por cero
    local instant_speed=$(( (current - last) * 60 / delta_time ))
    
    # Promedio móvil de últimos 3 intervalos
    SPEED_HISTORY+=($instant_speed)
    [ ${#SPEED_HISTORY[@]} -gt 3 ] && SPEED_HISTORY=("${SPEED_HISTORY[@]:1}")
    
    local total=0
    for s in "${SPEED_HISTORY[@]}"; do
        total=$((total + s))
    done
    
    [ ${#SPEED_HISTORY[@]} -eq 0 ] && echo 0 || echo $(( total / ${#SPEED_HISTORY[@]} ))
}

# Función para obtener progreso
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
echo -e "${BOLD}${RED}Presione Ctrl+C para salir${RESET}\n"

read INITIAL TOTAL < <(get_progress)
LAST_COUNT=$INITIAL

while true; do
    read CURRENT TOTAL < <(get_progress)
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    DELTA_TIME=$((CURRENT_TIME - LAST_UPDATE))
    DELTA_COUNT=$((CURRENT - LAST_COUNT))
    
    # Cálculo de velocidad y tiempo restante
    if [ $DELTA_COUNT -gt 0 ]; then
        SPEED=$(calculate_speed $CURRENT $LAST_COUNT $DELTA_TIME)
        REMAINING=$(( (TOTAL - CURRENT) * 60 / ($SPEED + 1) ))  # +1 para evitar división por cero
    else
        SPEED=0
        REMAINING=0
    fi
    
    # Actualizar pantalla cada 5s o con cambios
    if [ $DELTA_COUNT -gt 0 ] || [ $((CURRENT_TIME - LAST_UPDATE)) -ge 5 ]; then
        draw_enhanced_progress "$CURRENT" "$TOTAL" "$ELAPSED" "$SPEED" "$REMAINING"
        LAST_UPDATE=$CURRENT_TIME
        LAST_COUNT=$CURRENT
    fi
    
    # Verificar finalización
    if [ $CURRENT -ge $TOTAL ] && [ $TOTAL -gt 0 ]; then
        echo -e "\n\n${BOLD}${GREEN}✔ Proceso completado!${RESET}"
        echo -e "${BOLD}Tiempo total: ${YELLOW}$(date -u -d @$ELAPSED +%H:%M:%S)${RESET}"
        exit 0
    fi
    
    sleep 1
done