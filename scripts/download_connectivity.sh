#!/bin/bash

# Download the connectivity graphs from Matterport3DSimulator to the top-level directory of this project.
#     Repository source: https://github.com/peteanderson80/Matterport3DSimulator
#     connectivity dir.: https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity

curl -L wget https://api.github.com/repos/peteanderson80/Matterport3DSimulator/tarball | tar xz --wildcards "*/connectivity" --strip-components=1
