#!/bin/sh
# Gaudi crash watch: one compact report of anything that happened since the last run.
# Remembers the SEL entry count and the last boot id in $STATE so each run reports only what is new.
STATE=$HOME/tools/health/last_check.state
mkdir -p "$(dirname "$STATE")"
last_sel=0; last_boot=""
[ -f "$STATE" ] && . "$STATE"

boot_id=$(cat /proc/sys/kernel/random/boot_id)
sel_lines=$(sudo -n ipmitool sel elist 2>/dev/null)
sel_count=$(printf '%s\n' "$sel_lines" | grep -c '|')

echo "time: $(date '+%Y-%m-%d %H:%M %Z')  $(uptime | sed 's/.*up/up/')"
[ -n "$last_boot" ] && [ "$boot_id" != "$last_boot" ] && echo "REBOOTED since last check. Boots:" && journalctl --list-boots --no-pager | tail -3
new_pstore=$(sudo -n find /var/lib/systemd/pstore -mindepth 1 -maxdepth 1 -mmin -65 2>/dev/null)
if [ -n "$new_pstore" ]; then
    echo "NEW CRASH RECORD:" $new_pstore
    for entry in $new_pstore; do
        sudo -n find "$entry" -type f -exec cat {} + 2>/dev/null
    done | grep -aE "Bank [0-9]+:|Hardware Error\]: (PROCESSOR|Error [0-9])|panic" | grep -v pool | sort -u | head -6
fi
if [ "$sel_count" -gt "$last_sel" ]; then
    echo "NEW SEL ENTRIES ($((sel_count - last_sel))):"
    printf '%s\n' "$sel_lines" | tail -n $((sel_count - last_sel))
fi
hw=$(journalctl -k -b --no-pager | grep -ciE 'machine check|hardware error|mce:.*bank|AER:|DPC:|pcieport.*error')
echo "kernel HW/PCIe error lines this boot: $hw"
[ "$hw" -gt 0 ] && journalctl -k -b --no-pager | grep -iE 'machine check|hardware error|mce:.*bank|AER:|DPC:|pcieport.*error' | tail -5
$HOME/claude-gaudi/bin/claude-gaudi status 2>&1 | sed -n 2p
docker ps -a --format '{{.Names}} {{.Status}}' | grep -E 'glm53flash-r[01]-r4h8 '
echo "settings: cap=$(cat /sys/devices/system/cpu/intel_pstate/max_perf_pct)% stability=$(systemctl is-active cpu-stability) watchdog=$(sudo -n ipmitool mc watchdog get | grep Action | sed 's/.*: *//')"
sudo -n ipmitool sdr type Temperature 2>/dev/null | grep -E 'CPU[12] Temp' | cut -d'|' -f1,5 | tr -s ' ' | tr '\n' ' '; echo

printf 'last_sel=%s\nlast_boot=%s\n' "$sel_count" "$boot_id" > "$STATE"
