/**
 * Path Validation Utilities
 * Prevents path traversal attacks
 */

import * as path from 'path';
import * as fs from 'fs';

/**
 * Validate that a user-provided path stays within the allowed base directory
 * Throws error if path traversal detected
 */
export function validatePath(userPath: string, baseDir: string): string {
  // Resolve to absolute paths
  const resolvedBase = path.resolve(baseDir);
  const resolvedPath = path.resolve(baseDir, userPath);
  
  // Normalize to handle . and .. segments
  const normalizedBase = path.normalize(resolvedBase);
  const normalizedPath = path.normalize(resolvedPath);
  
  // Ensure path stays within baseDir
  if (!normalizedPath.startsWith(normalizedBase + path.sep) && normalizedPath !== normalizedBase) {
    throw new Error(`Path traversal detected: ${userPath}`);
  }
  
  return normalizedPath;
}

/**
 * Validate file path and check for symlinks
 * Throws error if symlink detected or path invalid
 */
export async function validateFilePath(userPath: string, baseDir: string): Promise<string> {
  const validPath = validatePath(userPath, baseDir);
  
  try {
    const stats = await fs.promises.lstat(validPath);
    
    // Reject symlinks
    if (stats.isSymbolicLink()) {
      throw new Error(`Symlink not allowed: ${userPath}`);
    }
    
    return validPath;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
      // File doesn't exist yet (OK for write operations)
      return validPath;
    }
    throw error;
  }
}

/**
 * Validate directory path
 * Throws error if path traversal detected
 */
export function validateDirectoryPath(userPath: string, baseDir: string): string {
  const validPath = validatePath(userPath, baseDir);
  
  // Ensure it's a directory (if exists)
  try {
    const stats = fs.statSync(validPath);
    if (!stats.isDirectory()) {
      throw new Error(`Not a directory: ${userPath}`);
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw error;
    }
    // Directory doesn't exist yet (OK for mkdir operations)
  }
  
  return validPath;
}

/**
 * Check if file size is within limit
 */
export async function validateFileSize(filePath: string, maxSize: number = 10 * 1024 * 1024): Promise<void> {
  const stats = await fs.promises.stat(filePath);
  
  if (stats.size > maxSize) {
    throw new Error(`File too large: ${stats.size} bytes (max: ${maxSize})`);
  }
}

/**
 * Check if file is binary
 */
export async function isBinaryFile(filePath: string): Promise<boolean> {
  const buffer = await fs.promises.readFile(filePath);
  
  // Check first 8KB for null bytes
  const checkLength = Math.min(buffer.length, 8000);
  for (let i = 0; i < checkLength; i++) {
    if (buffer[i] === 0) {
      return true;
    }
  }
  
  return false;
}
