---
title: "[C++] Smart Pointers - unique_ptr, shared_ptr, weak_ptr - from the implementation to their usage"
date: 2021-08-07 20:42:33
categories: [Programming Language, C++]
tags: [C++, Optimisation]
lang: en
mathjax:
---

When we develop our program or the system continues to grow as time goes by, **memory leakage** is usually a pain we suffer most. To militate against this problem, C++ has introduced smart pointer family - `unique_ptr`, `shared_ptr`, `weak_ptr`, defined in header `<memory>`, since C++11.

> Precisely, `unique_ptr` inherits most of characteristics from `auto_ptr`, which has been implemented since C++03. Yet, due to the lack of `move` semantics, it encounters significant limitation: 
```C++ =
auto_ptr<int> p (new int(1));
vector<auto_ptr<int>> vec;

vec.push_back(p);            // !!! error, because of Copy constructor
vec.push_back(std::move(p))  // Okay, with move semantics since C++11
```
> Note that `auto_ptr` is **deprecated** now and should not be used any more for safety.

<!--more-->

## A Toy Example

Let's use a simple example to illustrate how these smart pointers work:

```C++ =
class Obj {
public:
    Obj () : name("DEFAULT")  {
        cout << "Construct " << name << "\n";
    }

    Obj (string name) : name(name) {
        cout << "Construct " << name << "\n";        
    }

    ~Obj() {
        cout << "Destroy " << name << "\n";
    }

    void SayHi() {
        cout << "Hi " << name << "!\n";
    }
private:
    string name;
};
```

When the object is created or released, the corresponded message will be printed.

## Original Way with `new` and `delete`

Before utilising smart pointers, developers who want to allocate a new object might write the below code:

```C++ =
Obj* p0 = new Obj("0");
// ...
delete p0;  // forgetting this would lead to memory leak
```

We have to carefully ensure the created object will be released before leaving the procedure; otherwise leading to memory leakage. However, relying on smart pointers, the system is able to automatically destroy the allocated object based upon whether it is owned by any pointer.

## `unique_ptr`

`unique_ptr`, as its name suggests, owns an object uniquely. That means you cannot have the same memory owned by two or more `unique_ptr` objects; thus, it is also **not copyable**. When the `unique_ptr` goes out of scope, its holding object will be destroyed immediately.

### Code Snippet from `std::unique_ptr`

The implementation of `unique_ptr` in libstdc++ is worth mentioning for clarity (I have omitted the detail):[^1]

``` C++ =
template <typename _Tp, typename _Dp = default_delete<_Tp>>
class unique_ptr
{
public:
    // ...

    // Before move from rvalue,
    // releasing the ownership of the managed object if it owns a pointer
    unique_ptr& operator=(unique_ptr&& __u) noexcept
    {
        reset(__u.release());
        get_deleter() = std::forward<_Ep>(__u.get_deleter());
        return *this;
    }

    // Destroy the object if necessary, equivalent to reset()
    unique_ptr& operator=(nullptr_t) noexcept
    {
        // _M_t stores the data of this unique_ptr object
        reset();
        return *this;
    }

    // Release the ownership of any stored pointer,
    // and return the pointer to the managed object
    pointer release() noexcept
    {
        // _M_t stores the data of the managed object
        return _M_t.release();
    }

    // Destroy the object if necessary
    void reset(pointer __p = pointer()) noexcept
    {
        // _M_t stores the data of the managed object
        _M_t.reset(std::move(__p));
    }

    // Disable copy from lvalue.
    unique_ptr(const unique_ptr&) = delete;
    unique_ptr& operator=(const unique_ptr&) = delete;
};
```

Briefly, `unique_ptr` will release the current holding memory if necessary. Moreover, its copy constructor and copy assignment are disabled.

### Example 1: `unique_ptr` is unique and not copyable

```C++ =
int main()
{
    // unique_ptr<Obj> p1 (new Obj("1"));           // This is okay
    unique_ptr<Obj> p1 (make_unique<Obj>("1"));     // Recommend since C++14

    // unique_ptr<Obj> p2 = p1;    // ! error, not copyable
    p1->SayHi();

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct 1
Hi 1!


<<... END ...>>

Destroy 1
```

The allocated space held by `p1` is released automatically when its lifetime ends.

> [It is recommended to use `make_unique` to create `unique_ptr` objects.](#The-advantages-of-make_unique)

### Example 2: original holding object will be destroyed before setting to null

```C++ =
int main()
{
    unique_ptr<Obj> p1 (make_unique<Obj>("1"));

    p1->SayHi();

    p1 = nullptr;           // Okay, destroy 1 before setting to null
    // p1.reset(nullptr);   // Same result
    // p1.reset();          // Same result

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct 1
Hi 1!
Destroy 1


<<... END ...>>
```

`p1` has been destroyed before setting it as null. Using the method `reset()` can achieve the same result as well.


### Example 3: original holding object will be destroyed before move assignment

```C++ =
int main()
{
    unique_ptr<Obj> p1 (make_unique<Obj>("1"));
    unique_ptr<Obj> p2 (make_unique<Obj>("2"));

    p2 = std::move(p1);          // Okay, destroy 2 before move
    // p2.reset(p1.release());   // Same result, release the p1 ownership, then move to p2

    // p1->SayHi();     // ! error, p1 no longer valid
    p2->SayHi();

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**

```
Construct 1
Construct 2
Destroy 2
Hi 1!


<<... END ...>>

Destroy 1
```

Object `2` has been destroyed while moving object `1` into `p2`. Note that you have to release the ownership of object `1` with `release()` firstly before assigning it to the next owner.


## The Advantages of `make_unique` over `new` operator {#The-advantages-of-make_unique}

`make_unique` has been introduced since C++14, and its implementation can be briefly rewritten as:[^2]

```C++ =
template<typename T, typename... Args>
// for single objects
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args&&... args)
{ 
    return unique_ptr<T>(new T(std::forward<Args>(args)...)); 
}
```

For the most part, it makes no difference between either `new` or `make_unique`:

```C++ =
unique_ptr<int>(new int(1));
make_unique<int>(1);   // no difference
```

However, the former may go awry if it involves several arguments evaluation in a function call:[^3]

```C++ =
func(unique_ptr<int>(new int(1)), Foo());  // unsafe
func(make_unique<int>(1), Foo());          // exception safe
```

Before C++17, the order of function arguments evaluation is not defined.[^4] 

Consider the evaluation sequence below:

1. Create `int (1)` object  &emsp;  ***--- Success ! ***
2. Execute `Foo()`  &emsp;  ***--- Exception ! ***
3. Assign the created `int (1)` to a `unique_ptr` object &emsp;  ***--- Cannot be performed ! ***

If `new` has been processed without assigning the newly created object to `unique_ptr`, and exception happens in `Foo()`, then memory leakage occurs because there is no way for `unique_ptr` object to access the newly created object to destroy it.

Therefore, when you would like to create `unique_ptr` objects, `make_unique` is a better choice for exception safety. Even though it is not introduced in C++11, defining it manually is quite simple as aforementioned.

## Construct an Array of `unique_ptr` Objects

There are two ways to construct an array of `unique_ptr` objects as well.

```C++ =
int main()
{
    // unique_ptr<Obj []> pArray (new Obj[3]());        // This is okay. Its size is 3
    unique_ptr<Obj[]> pArray (make_unique<Obj[]>(3));   // Recommend

    for (int i=0; i<3; ++i)
        pArray[i].SayHi();

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}

```

**Output:**
```
Construct DEFAULT
Construct DEFAULT
Construct DEFAULT
Hi DEFAULT!
Hi DEFAULT!
Hi DEFAULT!


<<... END ...>>

Destroy DEFAULT
Destroy DEFAULT
Destroy DEFAULT
```

Compared to a single object, the parameter in `make_unique` is **the array size** instead of an initialiser. 

Similar to the single object, its array version can be briefly rewritten as:

```C++ =
template<typename T>
// for array objects
typename std::enable_if<std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(size_t num)
{
    return unique_ptr<T>(new typename std::remove_extent<T>::type[num]());
}
```

Note the subtle difference against a single object is the judgement of `is_array<T>::value`.


## `shared_ptr`

Unlike `unique_ptr`, the ownership of an object can be shared by multiple `shared_ptr` objects. The object will be destroyed once it is not owned by any `shared_ptr` objects. This information can be obtained by the method `use_count()`

### Code Snippet from `std::shared_ptr`

The implementation of `shared_ptr` in libstdc++ is worth mentioning for clarity (I have omitted the detail):[^5]

```C++ =
template<typename _Tp>
class shared_ptr : public __shared_ptr<_Tp>
{
    // ...
    friend class weak_ptr<_Tp>;
};
```

`shared_ptr` inherits from `__shared_ptr` class[^6], and you could observe `weak_ptr` is able to access the private members in `shared_ptr` objects. Actually, `shared_ptr` and `weak_ptr` indeed cooperate together from time to time which will be elaborated later.


```C++ =
template<typename _Tp, _Lock_policy _Lp>
class __shared_ptr : public __shared_ptr_access<_Tp, _Lp>
{
public:
    // ...

    // Default constructor where its initial value of pointer 
    // and reference count are null and 0
    constexpr __shared_ptr() noexcept
    : _M_ptr(0), _M_refcount()
    { }

    // If an exception is thrown this constructor has no effect.
    // _M_refcount will be assigned as a newly created object - __shared_count.
    // The count in __shared_count will be incremented
    template<typename _Yp, typename _Del, typename = _UniqCompatible<_Yp, _Del>>
    __shared_ptr(unique_ptr<_Yp, _Del>&& __r) : _M_ptr(__r.get()), _M_refcount()
    {
        auto __raw = __to_address(__r.get());
        _M_refcount = __shared_count<_Lp>(std::move(__r));
        _M_enable_shared_from_this_with(__raw);
    }

    // Similar to the constructor, the count in __shared_count will be incremented
    // and the current object will be replaced with the new one
    __shared_ptr& operator=(__shared_ptr&& __r) noexcept
    {
        __shared_ptr(std::move(__r)).swap(*this);
        return *this;
    }

    /// Return the number of owners
    long use_count() const noexcept
    {
        // _M_refcount is an object storing the reference counter information
        return _M_refcount._M_get_use_count();
    }

    // Construct a null shared ptr and and the current object will be replaced with it
    void reset() noexcept
    {
        __shared_ptr().swap(*this);
    }

    template<typename _Yp>
	_SafeConv<_Yp> reset(_Yp* __p) // _Yp must be complete.
	{
	  // Catch self-reset errors.
	  __glibcxx_assert(__p == 0 || __p != _M_ptr);
	  __shared_ptr(__p).swap(*this);
	}

private:
    // ...
    element_type*        _M_ptr;         // Contained pointer.
    __shared_count<_Lp>  _M_refcount;    // Reference counter.
};
```

Briefly, `shared_ptr` object records the number of owners (`__shared_count`) as well as the pointer of the managed object. Besides, `__shared_ptr_access`, the parent class, defines whether this object should be accessed by array operators or not.


```C++ =
template<_Lock_policy _Lp>
class __shared_count
{
public:
    // ...

    // Increase the reference counter
    __shared_count(const __shared_count& __r) noexcept
    : _M_pi(__r._M_pi)
    {
        if (_M_pi != 0)
            _M_pi->_M_add_ref_copy();
    }

    // Increase the reference counter
    __shared_count& operator=(const __shared_count& __r) noexcept
    {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != _M_pi)
        {
            if (__tmp != 0)
                __tmp->_M_add_ref_copy();

            if (_M_pi != 0)
                _M_pi->_M_release();

            _M_pi = __tmp;
        }

        return *this;
    }

private:
    // ...
    friend class __weak_count<_Lp>;

    // Define the operations of the stored pointer
    _Sp_counted_base<_Lp>*  _M_pi;
};
```

The `__shared_count` object is to manage the information of reference count. Here, `__weak_count` class is able to access the private members of `__shared_count`. Most of operations on the pointer itself are specified in `_Sp_counted_base`.

```C++ =
template<_Lock_policy _Lp = __default_lock_policy>
class _Sp_counted_base : public _Mutex_base<_Lp>{  /* ... */  };

template<>
inline void _Sp_counted_base<_S_single>::_M_release() noexcept
{
    if (--_M_use_count == 0)
    {
        // the managed object will be destroyed once it is not held by any shared_ptr
        _M_dispose();
        if (--_M_weak_count == 0)
            // the control block is not released until its weak_ptr count reaches zero
            _M_destroy();
    }
}
```

Note that while a `shared_ptr` object is created, the lifetime of the managed object and its **control block** might not be the same. The former is based upon whether it is shared by any `shared_ptr` objects, whereas the later is `weak_ptr`.

> Control block contains the information about how to allocate or deallocate the managed object, e.g. its allocator and deleter.

### Example 1: several `shared_ptr` can share the same object

```C++ =
int main()
{
    // shared_ptr<Obj> sp0 (new Obj("1"));           // Okay
    shared_ptr<Obj> sp0 = make_shared<Obj>("1");     // Recommend
    cout << "count = " << sp0.use_count() << "\n";

    shared_ptr<Obj> sp1 = sp0;
    cout << "count = " << sp0.use_count() << "\n";

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct 1
count = 1
count = 2


<<... END ...>>

Destroy 1
```

Object `1` can be held by both `sp0` and `sp1`. It is recommended to use `make_shared` to create the smart pointer due to [the same reason explained in `make_unique`](#The-advantages-of-make_unique).


### Example 2: the managed object will be destroyed when it is not held by any `shared_ptr`

```C++ =
int main()
{
    shared_ptr<Obj> sp2 = make_shared<Obj>("2");
    shared_ptr<Obj> sp3 = sp2;

    cout << "count = " << sp2.use_count() << "\n";

    sp3.reset();        // release the ownership from sp3
    //sp3 = nullptr;    // Same effect
    cout << "count = " << sp2.use_count() << "\n";

    sp2.reset();        // the reference count of object 2 becomes 0, destroyed
    cout << "count = " << sp2.use_count() << "\n";

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct 2
count = 2
count = 1
Destroy 2
count = 0


<<... END ...>>
```

Object `2` is destroyed while its reference count becomes 0.


## Construct an Array of `shared_ptr` Objects

Unlike `unique_ptr` objects, `shared_ptr` cannot dynamically allocate an array via `make_shared` until C++20. Moreover, prior to C++17, developers ought to specify the deleter to manage dynamically allocated arrays.[^7]

```C++ =
int main()
{
    // suggest to provide a deleter for array
    // so it uses delete[] to free the resoure instead of delete
    shared_ptr<Obj[]> spArray (new Obj[3], std::default_delete<Obj[]>());  // Okay, 
    //shared_ptr<Obj[]> spArray (make_shared<Obj[]>(3));  // Not available until C++20

    for (int i=0; i<3; ++i)
        spArray[i].SayHi();

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct DEFAULT
Construct DEFAULT
Construct DEFAULT
Hi DEFAULT!
Hi DEFAULT!
Hi DEFAULT!


<<... END ...>>

Destroy DEFAULT
Destroy DEFAULT
Destroy DEFAULT
```

> Actually, it is worth mentioning that since GCC 7.5, a default deleter to handle the dynamically allocated array has been provided by `struct __sp_array_delete`:
>
```C++ =
// The default deleter for shared_ptr<T[]> and shared_ptr<T[N]>.
struct __sp_array_delete
{
    template<typename _Yp>
    void operator()(_Yp* __p) const { delete[] __p; }
};
```
>
>It works as follows:
>
```C++ =
template<typename _Tp, _Lock_policy _Lp>
class __shared_ptr : public __shared_ptr_access<_Tp, _Lp>
{
    // ...

    // Constructor will check whether it is an array
    template<typename _Yp, typename = _SafeConv<_Yp>>
    explicit
    __shared_ptr(_Yp* __p)
    : _M_ptr(__p), _M_refcount(__p, typename is_array<_Tp>::type())
    {
        // ...
        _M_enable_shared_from_this_with(__p);
    }
};
```
>
>The `__shared_ptr` constructor checks whether the managed object is an array.
>
```C++ =
template<_Lock_policy _Lp>
class __shared_count
{
    // ...
    template<typename _Ptr>
    __shared_count(_Ptr __p, /* is_array = */ true_type)
    : __shared_count(__p, __sp_array_delete{}, allocator<void>())
    { }
};
```
>
>If the pointer is an array, the default deleter will be designated as `__sp_array_delete`.
>
>Thus, the below code should work fine even if we do not provide a deleter:
>
```C++ =
int main()
{
    shared_ptr<Obj[]> spArray (new Obj[3]);  // Okay if deleter is not provided
    // ...
    return 0;
}
```
>
> However, prior to C++17, we still suggest developers follow the rule to avoid undefined behavior resulted from compiler dependency.


## Leakage caused by Circular Reference within `shared_ptr` {#leakage-caused-by-circular-reference}

Consider the case when you would like to implement list data structure, the code might look like as follows:

```C++ =
struct List {
    int val;
    shared_ptr<List> next;

    List(int val) : val(val) {
        cout << "Construct node " << val << "\n";
    }

    ~List()
    {
        cout << "Destroy node " << val << "\n";
    }
};
```

And the list might form a cycle:

```C++ =
int main()
{
    shared_ptr<List> node_1 (make_shared<List>(1));
    shared_ptr<List> node_2 (make_shared<List>(2));

    node_1->next = node_2;
    node_2->next = node_1;      // A cycle is formed

    cout << "count = " << node_1.use_count() << "\n";

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```
Construct node 1
Construct node 2
count = 2


<<... END ...>>
```

Under this circumstance, both `node 1` and `node 2` are referenced to one another, the user count never degrades to zero even when the program terminates, leading to memory leakage. This problem is called **circular reference** or **cyclic dependency** issue.

To solve this problem, `weak_ptr` could come into play, which won't affect the reference count of an object. We may simply modify our list structure as follows:

```C++ =
struct List {
    int val;
    weak_ptr<List> next;      // weak reference

    List(int val) : val(val) {
        cout << "Construct node " << val << "\n";
    }

    ~List()
    {
        cout << "Destroy node " << val << "\n";
    }
};
```

**Output:**
```
Construct node 1
Construct node 2
count = 1


<<... END ...>>

Destroy node 2
Destroy node 1
```

Now, with the help of `weak_ptr`, the reference count does not increase for the reason that it is not considered an owner for this object. The memory leakage issue can be prevented.


## `weak_ptr`

`weak_ptr` can hold a "weak" reference to an object, which won't affect the reference count managed by `shared_ptr`. The main aim of `weak_ptr` is to own a temporary ownership so we can track this object. It is also an effective way to prevent [the memory leakage problem caused by a cycle within `shared_ptr` objects](#leakage-caused-by-circular-reference). In addition to that, we should be aware that even when the object held by `shared_ptr` is destroyed, the lifetime of its control block might be extended. To access the referenced object of `weak_ptr`, users should use the method `lock()` to get its original `shared_ptr` object first.


### Code Snippet from `std::weak_ptr`

Let's take a look at the implementation in libstdc++ (detail is omitted):[^8]

```C++ =
template<typename _Tp>
class weak_ptr : public __weak_ptr<_Tp>
{
    // ...

    // Get the shared_ptr object
    shared_ptr<_Tp> lock() const noexcept
    { 
        return shared_ptr<_Tp>(*this, std::nothrow);
    }
};
```

`weak_ptr` inherits from `__weak_ptr` and using `lock()` would help the user get a converted `shared_ptr` object.

```C++ =
template<typename _Tp, _Lock_policy _Lp>
class __weak_ptr
{
public:
    // ...

    // The default constructor would assign the pointer
    // and the reference count as null and 0 separately
    constexpr __weak_ptr() noexcept
    : _M_ptr(nullptr), _M_refcount()
    { }

    // assign the reference count managed by the shared_ptr object
    template<typename _Yp>
    _Assignable<_Yp> operator=(const __shared_ptr<_Yp, _Lp>& __r) noexcept
    {
        _M_ptr = __r._M_ptr;
        _M_refcount = __r._M_refcount;
        return *this;
    }

    // Get the reference count held by the shared_ptr object
    long use_count() const noexcept
    { 
        return _M_refcount._M_get_use_count();
    }

    // Equivelent to use_count() == 0,
    // meaning that the manged object has been deleted
    bool expired() const noexcept
    {
        return _M_refcount._M_get_use_count() == 0;
    }

    // Construct an empty weak_ptr and replace with it
    void reset() noexcept
    {
        __weak_ptr().swap(*this);
    }

 private:
    // ...
    element_type*      _M_ptr;        // Contained pointer.
    __weak_count<_Lp>  _M_refcount;   // Reference counter.
};
```

Briefly, `weak_ptr` can be used to track the managed object held by `shared_ptr`. It also has its own weak reference count.

```C++ =
template<_Lock_policy _Lp>
class __weak_count
{
public:
    // ...

    // Increase the "weak" reference counter
    __weak_count& operator=(const __shared_count<_Lp>& __r) noexcept
    {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != nullptr)
            __tmp->_M_weak_add_ref();

        if (_M_pi != nullptr)
            _M_pi->_M_weak_release();

        _M_pi = __tmp;
        return *this;
    }

    // Increase the "weak" reference counter
    __weak_count& operator=(const __weak_count& __r) noexcept
    {
        _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
        if (__tmp != nullptr)
            __tmp->_M_weak_add_ref();

        if (_M_pi != nullptr)
            _M_pi->_M_weak_release();

        _M_pi = __tmp;
        return *this;
    }
private:
    friend class __shared_count<_Lp>;

    // Define the operations of the stored pointer
    _Sp_counted_base<_Lp>*  _M_pi;
};
```

The mechanism of the reference counter in `weak_ptr` is quite similar to `shared_ptr`. Yet, `weak_ptr` mainly operates upon weak reference.

```C++ =
template<_Lock_policy _Lp = __default_lock_policy>
class _Sp_counted_base : public _Mutex_base<_Lp>
{
public:
    // ...

    // Called when _M_use_count drops to zero, to release the resources
    // managed by *this. (Can be redefined by the deriving class)
    virtual void _M_dispose() noexcept = 0;

    // Called when _M_weak_count drops to zero.
    // (Can be redefined by the deriving class)
    virtual void _M_destroy() noexcept
    { delete this; }
private:
    // ...
    _Atomic_word  _M_use_count;     // #shared
    _Atomic_word  _M_weak_count;    // #weak + (#shared != 0)
};
```

`_M_use_count` is used by `shared_ptr` to store the strong reference count, whereas `_M_weak_count` is used by `weak_ptr` to store the weak reference count.

```C++ =
template<>
inline void _Sp_counted_base<_S_single>::_M_release() noexcept
{
    if (--_M_use_count == 0)
    {
        // the managed object will be destroyed once it is not held by any shared_ptr
        _M_dispose();
        if (--_M_weak_count == 0)
            // the control block is not released until its weak_ptr count reaches zero
            _M_destroy();
    }
}

template<>
inline void _Sp_counted_base<_S_single>::_M_weak_release() noexcept
{
    // the control block is not released until its weak_ptr count reaches zero
    if (--_M_weak_count == 0)
        _M_destroy();
}
```

Once the weak reference count reaches zero, the control block will be deallocated.

### Example 1: `weak_ptr` does not increase the reference count held by `shared_ptr`

```C++ =
int main()
{
    shared_ptr<Obj> sp = make_shared<Obj>("1");
    cout << "count = " << sp.use_count() << "\n";

    weak_ptr<Obj> wp = sp;
    cout << "count = " << sp.use_count() << "\n";

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```C++ =
Construct 1
count = 1
count = 1


<<... END ...>>

Destroy 1
```

Aside from this, `wp.use_count()` returns the number of `shared_ptr` that manages this object; thus the value should be the same as `sp.use_count()`.

### Example 2: access the object through `lock()`

```C++ =
int main()
{
    shared_ptr<Obj> sp = make_shared<Obj>("1");
    weak_ptr<Obj> wp = sp;

    wp.lock()->SayHi();

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```C++ =
Construct 1
Hi 1!


<<... END ...>>

Destroy 1
```

Users should use `lock()` to get the `shared_ptr` pointer to access this object.



### Example 3: use `expired()` to check the availability of an object

```C++ =
int main()
{
    weak_ptr<Obj> wp;

    {
        shared_ptr<Obj> sp = make_shared<Obj>("1");
        wp = sp;
    }
    
    if (wp.expired())
        cout << "Object has been destroyed!" << "\n";

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**
```C++ =
Construct 1
Destroy 1
Object has been destroyed!


<<... END ...>>
```

`expired()` is equivelent to `use_count() == 0`.


### Example 4: release the reference from `weak_ptr`

```C++ =
int main()
{
    shared_ptr<Obj> sp = make_shared<Obj>("1");
    weak_ptr<Obj> wp = sp;

    //wp = nullptr;         // error
    wp.reset();

    //wp.lock()->SayHi();   // error, not available anymore

    cout << "\n\n<<... END ...>>\n\n";
    return 0;
}
```

**Output:**

```
Construct 1


<<... END ...>>

Destroy 1
```

You cannot assign a `weak_ptr` object as null directly.


## Construct an Array of `weak_ptr` Objects

You cannot construct an array of `weak_ptr` objects as the approach in `unique_ptr` and `shared_ptr` because the operator `[]` to access the array elements is not defined.

```C++ =
weak_ptr<Obj[]> wpArray;    // ??? wpArray[i] is not defined
```

A workaround is to declare it as a regular array, then tackle its elements separately:

```C++ =
weak_ptr<Obj> wpArray[3];

shared_ptr<Obj> sp1 = make_shared<Obj>("1");
shared_ptr<Obj> sp2 = make_shared<Obj>("2");
shared_ptr<Obj> sp3 = make_shared<Obj>("3");

wpArray[0] = sp1;
wpArray[1] = sp2;
wpArray[2] = sp3;
```


[^1]: [Source code of std::unique_ptr in GCC 11.2](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/api/a00371_source.html)

[^2]: [Source code of std::make_unique in GCC 11.2](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/api/a00371_source.html)

[^3]: [A proposal to add make_unique for symmetry, simplicity, and safety](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3588.txt)

[^4]: [C++17 - Wording for Order of Evaluation of Function Arguments](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0400r0.html)

[^5]: [Source code of std::shared_ptr in GCC 11.2](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/api/a17923_source.html)

[^6]: [Source code of __shared_ptr class in GCC 11.2](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/api/a00494_source.html)

[^7]: [shared_ptr to an array : should it be used?](https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used/13062069#13062069)

[^8]: [Source code of std::weak_ptr in GCC 11.2](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/libstdc++/api/a00494_source.html)