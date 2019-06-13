from copy import copy


class StructInfo(object):

    def __init__(self, attributes):
        self.attributes = tuple(attributes)


class Struct(object):
    __struct__ = StructInfo(())


    def __values__(self):
        attributes = self.__class__.__struct__.attributes
        values = [getattr(self, a) for a in attributes]
        return values

    def __names__(self):
        return self.__class__.__struct__.attributes

    def __build__(self, values):
        new_struct = copy(self)
        for val, attr in zip(values, self.__class__.__struct__.attributes):
            setattr(new_struct, attr, val)
        return new_struct

    def __flatten__(self):
        values = self.__values__()
        flat_list, recombine = _flatten_list(values)
        def recombine_self(flat_list):
            values = recombine(flat_list)
            return self.__build__(values)
        return flat_list, recombine_self

    @staticmethod
    def values(struct):
        if isinstance(struct, (list, tuple)):
            return struct
        if isinstance(struct, Struct):
            return struct.__values__()
        raise ValueError("Not a struct: %s" % struct)

    # @staticmethod
    # def names(struct):
    #     if isinstance(struct, (list, tuple)):
    #         return ['[%d]' % i for i in range(len(struct))]
    #     if isinstance(struct, Struct):
    #         return struct.__names__()
    #     raise ValueError("Not a struct: %s" % struct)


    @staticmethod
    def build(values, source_struct):
        if isinstance(source_struct, list):
            return list(values)
        if isinstance(source_struct, tuple):
            return tuple(values)
        if isinstance(source_struct, Struct):
            return source_struct.__build__(values)
        raise ValueError("Not a struct: %s" % source_struct)

    @staticmethod
    def flatten(struct):
        if isinstance(struct, Struct):
            return struct.__flatten__()
        if isinstance(struct, (tuple, list)):
            return _flatten_list(struct)
        return [struct], lambda tensors: tensors[0]

    @staticmethod
    def isstruct(object):
        return isinstance(object, (Struct, list, tuple))

    @staticmethod
    def map(f, struct):
        values = Struct.values(struct)
        values = [f(element) for element in values]
        return Struct.build(values, struct)

    @staticmethod
    def flatmap(f, struct):
        values, recombine = Struct.flatten(struct)
        values = [f(element) for element in values]
        return recombine(values)



def _flatten_list(struct_list):
    tensor_counts = []
    recombiners = []
    values = []
    for struct in struct_list:
        tensors, recombine = Struct.flatten(struct)
        values += tensors
        tensor_counts.append(len(tensors))
        recombiners.append(recombine)

    def recombine(tensor_list):
        new_structs = []
        for i in range(len(struct_list)):
            tensors = tensor_list[:tensor_counts[i]]
            struct = recombiners[i](tensors)
            new_structs.append(struct)
            tensor_list = tensor_list[tensor_counts[i]:]
        assert len(tensor_list) == 0, "Not all tensors were used in reassembly"
        if isinstance(struct_list, list):
            return new_structs
        else:
            return tuple(new_structs)

    return values, recombine





# def attributes(struct, remove_prefix=True, qualified_names=True):
#     array, reassemble = Struct.disassemble(struct)
#     ids = ["id%d" % i for i in range(len(array))]
#     id_struct = reassemble(ids)
#     _recursive_attributes(id_struct, ids, remove_prefix, qualified_names, None)
#     return id_struct
#
#
# def _recursive_attributes(struct, ids, remove_prefix, qualified_names, qualifier):
#     if not Struct.isstruct(struct): return
#
#     if isinstance(struct, (tuple,list)):
#         for entry in struct:
#             _recursive_attributes(entry, ids, remove_prefix, qualified_names, qualifier)
#     else:  # must be a Struct instance
#         for attr, val in struct.__dict__.items():
#             name = attr
#             if remove_prefix and name.startswith('_'):
#                 name = name[1:]
#             if qualified_names:
#                 if qualifier is None: qualified_name = name
#                 else: qualified_name = qualifier + '.' + name
#             else:
#                 qualified_name = name
#             if val in ids:
#                 setattr(struct, attr, qualified_name)
#             else:
#                 _recursive_attributes(val, ids, remove_prefix, qualified_names, qualified_name)


# class StructAttributeGetter(object):
#
#     def __init__(self, getter):
#         self.getter = getter
#
#     def __call__(self, struct):
#         return self.getter(struct)
#
#
# def selector(struct):
#     array, reassemble = disassemble(struct)
#     ids = ['#ref' % i for i in range(len(array))]
#     tagged_struct = reassemble(ids)
#     _recursive_selector(tagged_struct)
#     return tagged_struct
#
#
# def _recursive_selector(struct):
#     pass