#!/bin/sh

REMOTE_IP="$(echo $SSH_CLIENT | awk '{print $1}')"
tmux new-window -k -d -t comma -n remotebridge "/data/openpilot/cereal/messaging/bridge $REMOTE_IP customReservedRawData1"