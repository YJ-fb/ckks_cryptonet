#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Helper libraries
import os
import shutil

    
def removeDir(path = ''):
        """ Remove the directory at the given path """
        try:
            shutil.rmtree(path)
        except OSError as error:
            raise error

    
def createDir(path = '', force = False):
        """Create the directory at the given path. 
        
        Keyword arguments:
        path  -- the path of the directory
        force -- if true than the directory is removed first (default False)
        """
        
        if(force):
            try:
                removeDir(path)
            except OSError:
                pass
        try:
            os.makedirs(path)
            return True
        except OSError:
            return False