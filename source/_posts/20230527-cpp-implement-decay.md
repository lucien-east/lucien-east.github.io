---
title: "[C++] Implement std::decay from scratch"
date: 2023-05-27 08:52:20
categories: [Programming Language, C++]
tags: [C++]
lang: en
mathjax:
---

Since C++11, `std::decay` is introduced along with `<type_traits>`. It is used to _decay_ a type, or to convert a type into its corresponding **by-value** type. It will remove any top-level cv-qualifiers(`const`, `volatile`) and reference qualifiers for the specified type. For example, `int&` is turned into `int` and an array type becomes a pointer to its element types. Knowing its usage, we could try to implement our own version of `std::decay`.

<!--more-->

For `std::decay<T>`, the transformation of type `T` contains following parts:

- Removing references
- Removing cv-qualifiers (`const` and `volatile`)
- For an array type, yielding a pointer to its element type
- For a function type, yiedling its function pointer type


## Removing References

Firstly, we implement `RemoveReferenceT` trait to remove references:

```C++ =
template <typename T>
struct RemoveReferenceT {
    using Type = T;
};

// remove lvalue reference
template <typename T>
struct RemoveReferenceT<T&> {
    using Type = T;
};

// remove ravlue reference
template <typename T>
struct RemoveReferenceT<T&&> {
    using Type = T;
};

// alias for usage convenience
template <typename T>
using RemoveReference = typename RemoveReferenceT<T>::Type;
```

Results:

```C++
RemoveReference<int>          // int
RemoveReference<int&>         // int
RemoveReference<int&&>        // int
RemoveReference<const int>    // const int
RemoveReference<const int&>   // const int
```

The corresponding type trait in C++ STL is [std::remove_reference](https://en.cppreference.com/w/cpp/types/remove_reference)


## Removing cv-qualifiers

Then, `RemoveConstT` and `RemoveVolatileT` are to remove `const` and `volatile` qualifiers, respectively:

```C++ =
template <typename T>
struct RemoveConstT {
	using Type = T;
};

// remove const
template <typename T>
struct RemoveConstT<const T> {
	using Type = T;
};

// alias for usage convenience
template <typename T>
using RemoveConst = typename RemoveConstT<T>::Type;
```

```C++ =
template <typename T>
struct RemoveVolatileT {
	using Type = T;
};

// remove volatile
template <typename T>
struct RemoveVolatileT<volatile T> {
	using Type = T;
};

// alias for usage convenience
template <typename T>
using RemoveVolatile = typename RemoveVolatileT<T>::Type;
```

`RemoveConstT` and `RemoveVolatileT` can be composed into `RemoveCVT`:

```C++ =
// metafunction forwarding: inherit the Type member from RemoveConstT
template <typename T>
struct RemoveCVT : RemoveConstT<RemoveVolatile<T>> {};

// alias for usage convenience
template <typename T>
using RemoveCV = typename RemoveCVT<T>::Type;
```

Results:

```C++
RemoveCV<int>                  // int
RemoveCV<const int>            // int
RemoveCV<volatile int>         // int
RemoveCV<const volatile int>   // int

RemoveCV<const volatile int*>  // const volatile int*
RemoveCV<int* const volatile>  // int*
```

The corresponding type traits in C++ STL: [std::remove_cv, std::remove_const, std::remove_volatile](https://en.cppreference.com/w/cpp/types/remove_cv)


> Note that `const volatile int*` is not changed because the pointer itself is neither const or volatile. (See [const and volatile pointers](https://learn.microsoft.com/en-us/cpp/cpp/const-and-volatile-pointers?view=msvc-170))



With `RemoveReference` and `RemoveCVT` traits above, we can get a decay trait for nonarray and nonfunction cases:

```C++ =
// remove reference firstly and then cv-qualifier
template <typename T>
struct DecayT : RemoveCVT<RemoveReference<T>> {};
```

We name our version `DecayT` in order not to confuse with original `std::decay`.


## Array-to-pointer Decay

Now we take array types into account. Below are partial specialisations to convert an array type into a pointer to its element type:

```C++ =
// unbounded array
template <typename T>
struct DecayT<T[]> {
	using Type = T*;
};

// bounded array
template <typename T, std::size_t N>
struct DecayT<T[N]> {
	using Type = T*;
};
```


Similarly, C++ STL provides [std::is_array](https://en.cppreference.com/w/cpp/types/is_array) to check whether `T` is an array type.


## Function-to-pointer Decay

We want to recognise a function regardless of its return type and parameter types, and then get its function pointer. Because there are different number of parameters, we need to employ variadic templates:

```C++ =
template <typename Ret, typename...Args>
struct DecayT<Ret(Args...)> {
	using Type = Ret(*)(Args...);
};

// specialisation for variadic function
template <typename Ret, typename...Args>
struct DecayT<Ret(Args..., ...)> {
	using Type = Ret(*)(Args..., ...);
};
```

C++ STL also provides [std::is_function](https://en.cppreference.com/w/cpp/types/is_function) to check the function type.

> It is worth mentioning that many compilers nowadays use fundamental properties to check a function type for better performance instead[^1]:
>```C++
!std::is_const<const T>::value && !std::is_reference<T>::value
> ```
>
> - Functions are not objects; thus, `const` cannot be applied
> - When `const T` fails to be a const-qualified type, `T` is either a function type or a reference type
> - We can rule out reference types to get only with function types for `T`


Now, with alias template for convenience, we could get our own version of decay trait, `Decay`:

```C++ =
template <typename T>
using Decay = typename DecayT<T>::Type;
```

Results:

```C++
Decay<int&>         // int
Decay<const int>    // int
Decay<int const&>   // int
Decay<int[]>        // int*
Decay<int[3]>       // int*
Decay<int[3][2]>    // int*
Decay<int(int)>     // int(*)(int)
```


## In Comparison with `std::decay`

In fact, [C++ standard defines std::decay](https://timsong-cpp.github.io/cppwp/n4659/meta.trans.other#tab:type-traits.other) as:


| Template      | Comments |
| ----------- | ----------- |
| `template <class T>
struct decay;`      | Let U be `remove_­reference_­t<T>`. If `is_­array_­v<U>` is true, the member typedef type shall equal `remove_­extent_­t<U>*`. If `is_­function_­v<U>` is true, the member typedef type shall equal `add_­pointer_­t<U>`. Otherwise the member typedef type equals `remove_­cv_­t<U>`. [ Note: This behavior is similar to the lvalue-to-rvalue, array-to-pointer, and function-to-pointer conversions applied when an lvalue expression is used as an rvalue, but also strips cv-qualifiers from class types in order to more closely model by-value argument passing.  — end note ]   |


Most compilers directly follow the comments to implement the decay trait. Our own version in this article is basically a step-by-step implementation mentioned in the note for pedagogical purposes.


[^1]: [How is std::is_function implemented?](https://stackoverflow.com/questions/59654482/how-is-stdis-function-implemented)