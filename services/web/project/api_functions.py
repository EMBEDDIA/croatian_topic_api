import shutil
import psutil

from flask import redirect

#from rest_framework.response import Response

# class MLPHealthView(generics.GenericAPIView):
#     """
#     MLP health view. Displays information about service's health.
#     """


#     def _get_active_tasks(self):
#         """
#         Gets the number of active (running + queued) from message broker.
#         """
#         active_and_scheduled_tasks = 0
#         inspector = Inspect()
#         if inspector.connection is not None:
#             active_tasks = inspector.active()
#             scheduled_tasks = inspector.scheduled()
#             if active_tasks:
#                 active_and_scheduled_tasks += sum([len(tasks) for tasks in active_tasks.values()])
#             if scheduled_tasks:
#                 active_and_scheduled_tasks += sum([len(tasks) for tasks in scheduled_tasks.values()])
#             return active_and_scheduled_tasks

#         else:
#             return 0



def health():
    disk_total, disk_used, disk_free = shutil.disk_usage("/")
    memory = psutil.virtual_memory()

    mlp_health = {
        'service': 'comment_api',
        'version': '1.0',
#        'version': get_version(),
#        'loaded_languages': global_mlp_instance.stanford_pipelines.keys(),
        'disk': {
            'free': disk_free / (2 ** 30),
            'total': disk_total / (2 ** 30),
            'used': disk_used / (2 ** 30),
            'unit': 'GB'
        },
        'memory': {
            'free': memory.available / (2 ** 30),
            'total': memory.total / (2 ** 30),
            'used': memory.used / (2 ** 30),
            'unit': 'GB'
        },
        'cpu': {
            'percent': psutil.cpu_percent()
        },
#        'redis': check_redis_connection()
    }

    # gpu_count = torch.cuda.device_count()
    # gpu_devices = [torch.cuda.get_device_name(i) for i in range(0, gpu_count)]

    # mlp_health["gpu_usage_enabled"] = USE_GPU_FOR_STANFORD
    # mlp_health['active_tasks'] = self._get_active_tasks()
    # mlp_health['gpu'] = {
    #     'count': gpu_count,
    #     'devices': gpu_devices
    # }

    return mlp_health
    #return Response(mlp_health, status=status.HTTP_200_OK)

def documentation():
    return redirect('/swagger.json')
