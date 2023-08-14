#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <dirent.h>
#include <string.h>

#define N_PROCESS 50

char *input_dir = NULL;
char *output_dir = NULL;
char *binary = NULL;

void spwn_fuzzer(char *seed_path, char *output_path, char *tmpdir) {
  char *_args[] = {
    "/home/xuhang/GFuzz/Generator/afl-fuzz",
    "-i", seed_path,
    "-o", tmpdir,
    "-K", output_path,
    "--",
    binary,
    NULL
  };
  int nullfd = open("/dev/null", O_WRONLY);
  dup2(nullfd, 1);
  execv("/home/xuhang/GFuzz/Generator/afl-fuzz", _args);
}

int main(int argc, char **argv) {
  int opt = 0;
  int status = 0;
  struct dirent *ent;
  int rdfd = open("/dev/urandom", O_RDONLY);
  while (
      (opt = getopt(
           argc, argv,
           "I:O:B:")) >
      0) {

    switch (opt) {
      case 'I':
        input_dir = optarg;
        break;
      case 'O':
        output_dir = optarg;
        break;
      case 'B':
	binary = optarg;
	break;
    }
  }
  DIR *pdir = opendir(input_dir);
  char *tmp_dirs[N_PROCESS] = {NULL};
  int *pids[N_PROCESS];
  for (size_t i = 0; i < N_PROCESS; i ++) {
    pids[i] = -1;
  }
  while ((ent = readdir(pdir)) != NULL) {
    if (ent->d_type == DT_REG) {
      char path[4096] = {0};
      snprintf(path, 4095, "%s/%s/in", output_dir, ent->d_name);
      if (!access(path, F_OK)) {
        continue;
      }
      printf("[*] try to evaluate %s\n", ent->d_name);
    }
    else {
      continue;
    }
    int free_process = -1;
    for (size_t i = 0; i < N_PROCESS; i++) {
      if (pids[i] == -1) {
        free_process = i;
        break;
      }
    }
    if (free_process == -1) {
      printf("[*] wait for a fuzzing evaluate process to exit\n");
      pid_t exited_pid = waitpid(0, &status, 0);
      for (size_t i = 0; i < N_PROCESS; i++) {
        if (pids[i] == exited_pid) {
          free_process = i;
          char *cmd = calloc(1, 0x1000 * 2);
          snprintf(cmd, 0x1000 * 2 - 1, "rm -rf %s", tmp_dirs[i]);
          system(cmd);
          snprintf(cmd, 0x1000 * 2 - 1, "mkdir -p %s/%s/in", output_dir, ent->d_name);
          system(cmd);
          snprintf(cmd, 0x1000 * 2 - 1, "cp %s/%s %s/%s/in", input_dir, ent->d_name, output_dir, ent->d_name);
          printf("%s\n", cmd);
          system(cmd);
          free(cmd);
          free(tmp_dirs[i]);
          tmp_dirs[i] = NULL;
          pids[i] = -1;
        }
      }
    }
    else {
      char *cmd = calloc(1, 0x1000 * 2);
      snprintf(cmd, 0x1000 * 2 - 1, "mkdir -p %s/%s/in", output_dir, ent->d_name);
      system(cmd);
      snprintf(cmd, 0x1000 * 2 - 1, "cp %s/%s %s/%s/in", input_dir, ent->d_name, output_dir, ent->d_name);
      system(cmd);
      free(cmd);
    }

    char *tmp_dir = calloc(12, 1);
    memcpy(tmp_dir, "/tmp/", 5);
    while (1) {
      for (size_t i = 5; i < 11; i++) {
        read(rdfd, &tmp_dir[i], 1);
        tmp_dir[i] = (unsigned char)tmp_dir[i] % 26 + 'A';
      }
      if (access(tmp_dir, F_OK)) {
        break;
      }
    }

    tmp_dirs[free_process] = tmp_dir;
    int cpid = fork();
    if (!cpid) {
      char *seed_path = calloc(1, 0x1000);
      snprintf(seed_path, 0x1000, "%s/%s/in", output_dir, ent->d_name);
      char *output_path = calloc(1, 0x1000);
      snprintf(output_path, 0x1000, "%s/%s/", output_dir, ent->d_name);
      mkdir(output_dir, 0766);
      spwn_fuzzer(seed_path, output_path, tmp_dirs[free_process]);
      exit(-1);
    }
    printf("[+] seed %s bind to pid %d\n", ent->d_name, cpid);
    pids[free_process] = cpid;
    tmp_dir = NULL;
  }
  int wpid = -1;
  while(wpid = wait(&status) > 0);

  return 0;
}

